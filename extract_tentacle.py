#!/usr/bin/env python3
"""Extract English DOTT Remastered voice samples and subtitle text."""

import argparse
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "vendor"))

import fsb5  # noqa: E402


DEFAULT_ARCHIVE = BASE_DIR / "source" / "tenta.cle"
SPEECH_INFO_ENTRY_RE = re.compile(
    rb"([A-Z0-9]{3}_[A-Za-z0-9_.-]+_[0-9]+_[0-9]+)\0[^\0]*\0"
)


class FileView:
    """Seekable bounded view over one member of the LPAK archive."""

    def __init__(self, stream, start, length):
        self.stream = stream
        self.start = start
        self.length = length
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.length + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if position < 0 or position > self.length:
            raise ValueError("Seek outside archive member")
        self.position = position
        self.stream.seek(self.start + position)
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        self.stream.seek(self.start + self.position)
        data = self.stream.read(size)
        self.position += len(data)
        return data


def read_index(stream):
    stream.seek(0)
    header = stream.read(32)
    if len(header) != 32 or header[:4] != b"KAPL":
        raise ValueError("Not a supported Day of the Tentacle LPAK archive")
    version = struct.unpack_from("<f", header, 4)[0]
    if version != 1.0:
        raise ValueError(f"Unsupported LPAK version: {version}")

    entries_offset, names_offset, data_offset, _, entries_length = (
        struct.unpack_from("<5I", header, 12)
    )
    if entries_length % 20:
        raise ValueError("Invalid LPAK entry table length")
    entry_count = entries_length // 20

    names = []
    stream.seek(names_offset)
    for _ in range(entry_count):
        name = bytearray()
        while True:
            char = stream.read(1)
            if char == b"\0":
                break
            if not char:
                raise ValueError("Unexpected end of LPAK name table")
            name.extend(char)
        names.append(name.decode("utf-8"))

    entries = {}
    for index, name in enumerate(names):
        stream.seek(entries_offset + index * 20)
        relative_offset, _, stored_size, unpacked_size, compressed = struct.unpack(
            "<5I", stream.read(20)
        )
        entries[name] = (
            relative_offset,
            stored_size,
            unpacked_size,
            compressed,
        )
    return data_offset, entries


def open_member(stream, data_offset, entries, name):
    try:
        relative_offset, stored_size, unpacked_size, compressed = entries[name]
    except KeyError as error:
        raise ValueError(f"Archive member not found: {name}") from error
    if compressed or stored_size != unpacked_size:
        raise ValueError(f"Compressed archive member is not supported: {name}")
    return FileView(stream, data_offset + relative_offset, unpacked_size)


def read_voice_sample_names(stream, data_offset, entries):
    bank = fsb5.FSB5(
        open_member(stream, data_offset, entries, "audio/iMUSEClient_VO.fsb")
    )
    if bank.get_sample_extension() != "mp3":
        raise ValueError("Expected MPEG/MP3 voice samples")
    return [sample.name for sample in bank.samples]


def find_subtitle(info, sample_name):
    results = find_subtitle_values(info, sample_name)
    if not results:
        return None
    if len(results) != 1:
        raise ValueError(f"Conflicting subtitle entries for {sample_name}")
    return results[0]


def find_subtitle_values(info, sample_name):
    key = sample_name.removeprefix("EN_").encode("ascii") + b"\0"
    results = []
    position = 0
    while True:
        position = info.find(key, position)
        if position < 0:
            break
        start = position + len(key)
        end = info.find(b"\0", start)
        if end < 0:
            raise ValueError(f"Unterminated subtitle entry for {sample_name}")
        value = info[start:end]
        if value not in results:
            results.append(value)
        position += len(key)
    if not results:
        return []
    decoded = []
    for value in results:
        try:
            decoded.append(value.decode("utf-8"))
        except UnicodeDecodeError:
            decoded.append(value.decode("cp1252"))
    return decoded


def find_orphan_subtitle_merges(info, sample_names, all_sample_names=None):
    subtitle_keys = {
        match.group(1).decode("ascii")
        for match in SPEECH_INFO_ENTRY_RE.finditer(info)
    }
    sample_by_key = {}
    audio_groups = defaultdict(list)
    for sample_name in sample_names:
        if not sample_name.startswith("EN_"):
            raise ValueError(f"Expected English sample name: {sample_name}")
        key = sample_name[3:]
        if key in sample_by_key:
            raise ValueError(f"Duplicate English sample name: {sample_name}")
        sample_by_key[key] = sample_name
        match = re.fullmatch(r"(.+)_([0-9]+)", key)
        if match:
            audio_groups[match.group(1)].append((int(match.group(2)), key))

    existing_sample_keys = {
        name[3:] if name.startswith(("EN_", "DE_")) else name
        for name in (all_sample_names if all_sample_names is not None else sample_names)
    }
    orphan_groups = defaultdict(list)
    for key in subtitle_keys - existing_sample_keys:
        match = re.fullmatch(r"(.+)_([0-9]+)", key)
        if match:
            orphan_groups[match.group(1)].append((int(match.group(2)), key))

    merges = []
    for prefix, voiced_samples in sorted(audio_groups.items()):
        voiced_samples.sort()
        orphan_samples = sorted(orphan_groups.get(prefix, ()))
        for (previous_number, previous_key), (next_number, next_key) in zip(
            voiced_samples, voiced_samples[1:]
        ):
            for orphan_number, orphan_key in orphan_samples:
                if not previous_number < orphan_number < next_number:
                    continue
                orphan_text = find_subtitle(info, "EN_" + orphan_key)
                if not orphan_text or not orphan_text.strip():
                    continue
                merges.append(
                    {
                        "audio": sample_by_key[previous_key],
                        "subtitle_key": orphan_key,
                        "next_audio": sample_by_key[next_key],
                    }
                )
        if voiced_samples:
            last_number, last_key = voiced_samples[-1]
            for orphan_number, orphan_key in orphan_samples:
                if orphan_number <= last_number:
                    continue
                orphan_text = find_subtitle(info, "EN_" + orphan_key)
                if not orphan_text or not orphan_text.strip():
                    continue
                merges.append(
                    {
                        "audio": sample_by_key[last_key],
                        "subtitle_key": orphan_key,
                        "next_audio": None,
                    }
                )
    return merges


def assigned_subtitle_keys(sample_names, merges):
    assignments = {
        sample_name: [sample_name.removeprefix("EN_")]
        for sample_name in sample_names
    }
    for merge in merges:
        try:
            assignments[merge["audio"]].append(merge["subtitle_key"])
        except KeyError as error:
            raise ValueError(
                f"Subtitle merge refers to unknown audio sample: {merge['audio']}"
            ) from error
    return assignments


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="path to tenta.cle",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "audio",
        help="output directory (default: audio beside this script)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace matching MP3 and TXT outputs",
    )
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")

    args.out.mkdir(parents=True, exist_ok=True)
    with args.archive.open("rb") as archive:
        data_offset, entries = read_index(archive)
        speech_info = open_member(
            archive, data_offset, entries, "localization/en.speech.info"
        ).read()
        voice_bank = fsb5.FSB5(
            open_member(archive, data_offset, entries, "audio/iMUSEClient_VO.fsb"),
            prefix="EN_",
        )
        if voice_bank.get_sample_extension() != "mp3":
            raise ValueError("Expected MPEG/MP3 voice samples")
        samples = [sample for sample in voice_bank.samples if sample.name.startswith("EN_")]
        if not samples:
            raise ValueError("No English voice samples found")
        sample_names = [sample.name for sample in samples]
        subtitle_merges = find_orphan_subtitle_merges(
            speech_info,
            sample_names,
            [sample.name for sample in voice_bank.samples],
        )
        subtitle_keys = assigned_subtitle_keys(sample_names, subtitle_merges)

        targets = []
        for sample in samples:
            stem = safe_filename(sample.name)
            targets.extend((args.out / f"{stem}.mp3", args.out / f"{stem}.txt"))
        if not args.overwrite:
            existing = [path for path in targets if path.exists()]
            if existing:
                raise FileExistsError(
                    f"{len(existing)} output files already exist; use --overwrite"
                )

        rebuild = getattr(voice_bank, "rebuild_sample", None)
        if rebuild is None:
            rebuild = voice_bank._rebuild_sample

        exact = sfx = unresolved = 0
        for sample in samples:
            stem = safe_filename(sample.name)
            subtitle_parts = []
            for subtitle_key in subtitle_keys[sample.name]:
                text = find_subtitle(speech_info, "EN_" + subtitle_key)
                if text and text.strip():
                    subtitle_parts.append(text.strip())
            subtitle = " ".join(subtitle_parts) or None
            if subtitle is None and "_sfx_" in sample.name.lower():
                subtitle = "[Non-speech sound; no subtitle in the English index]"
                sfx += 1
            elif subtitle is None:
                subtitle = "[No matching English subtitle in the game index]"
                unresolved += 1
            else:
                exact += 1

            audio_path = args.out / f"{stem}.mp3"
            text_path = args.out / f"{stem}.txt"
            audio_mode = "wb" if args.overwrite else "xb"
            text_mode = "w" if args.overwrite else "x"
            with audio_path.open(audio_mode) as audio_file:
                audio_file.write(rebuild(sample))
            with text_path.open(text_mode, encoding="utf-8", newline="") as text_file:
                text_file.write(subtitle + "\r\n")

    print(
        f"Exported {len(samples)} English MP3/TXT pairs to {args.out} "
        f"(indexed subtitles: {exact}; SFX markers: {sfx}; unresolved: {unresolved})."
    )
    print(
        f"Attached {len(subtitle_merges)} subtitle-only fragment(s) "
        "to the preceding voice sample."
    )


if __name__ == "__main__":
    main()
