#!/usr/bin/env python3
"""Replace German speech and its Spanish menu label in tenta.cle."""

import argparse
import csv
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "source" / "tenta.cle"
DEFAULT_AUDIO_DIR = ROOT / "audio_es"
DEFAULT_OUTPUT = ROOT / "source" / "tenta_es.cle"
FSB_MEMBER = "audio/iMUSEClient_VO.fsb"
UITEXT_MEMBER = "localization/sp.uitext.info"
SAMPLE_RATE = 48000
BITRATE = 320000
FSB_ALIGNMENT = 32
COPY_BUFFER_SIZE = 1024 * 1024

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import extract_tentacle  # noqa: E402
import fsb5  # noqa: E402


@dataclass(frozen=True)
class LpakEntry:
    index: int
    name: str
    raw: bytes
    relative_offset: int
    stored_size: int
    unpacked_size: int
    compression: int


def copy_range(source, destination, start, length):
    source.seek(start)
    remaining = length
    while remaining:
        block = source.read(min(COPY_BUFFER_SIZE, remaining))
        if not block:
            raise EOFError(
                f"Unexpected end of input while copying {length} bytes at {start}"
            )
        destination.write(block)
        remaining -= len(block)


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def original_sample_size(samples, data_size, index):
    sample = samples[index]
    end = samples[index + 1].dataOffset if index + 1 < len(samples) else data_size
    return end - sample.dataOffset


def read_lpak_entries(stream):
    data_offset, indexed_entries = extract_tentacle.read_index(stream)
    stream.seek(0)
    header = stream.read(32)
    if len(header) != 32:
        raise ValueError("Truncated LPAK header")
    entries_offset, _, header_data_offset, _, entries_length = struct.unpack_from(
        "<5I", header, 12
    )
    if data_offset != header_data_offset or entries_length % 20:
        raise ValueError("Invalid LPAK entry table")
    names = list(indexed_entries)
    if len(names) != entries_length // 20:
        raise ValueError("LPAK entry count does not match its name index")

    entries = []
    for index, name in enumerate(names):
        stream.seek(entries_offset + index * 20)
        raw = stream.read(20)
        if len(raw) != 20:
            raise ValueError(f"Truncated LPAK entry {name}")
        relative_offset, _, stored_size, unpacked_size, compression = struct.unpack(
            "<5I", raw
        )
        indexed = indexed_entries[name]
        if indexed != (relative_offset, stored_size, unpacked_size, compression):
            raise ValueError(f"Inconsistent LPAK index data for {name}")
        if compression or stored_size != unpacked_size:
            raise ValueError(
                f"Compressed LPAK member is not supported: {name}"
            )
        entries.append(
            LpakEntry(
                index,
                name,
                raw,
                relative_offset,
                stored_size,
                unpacked_size,
                compression,
            )
        )

    previous_end = 0
    for entry in sorted(entries, key=lambda item: item.relative_offset):
        if entry.relative_offset < previous_end:
            raise ValueError(f"Overlapping LPAK member: {entry.name}")
        previous_end = entry.relative_offset + entry.stored_size
    if stream.seek(0, os.SEEK_END) < data_offset + previous_end:
        raise ValueError("LPAK member extends beyond end of archive")
    return header, entries_offset, data_offset, entries


def normalize_sample_key(value):
    key = value.strip()
    if key.endswith(".es.wav"):
        key = key[: -len(".es.wav")]
    if key.startswith(("DE_", "EN_")):
        key = key[3:]
    if not key:
        raise ValueError("Empty sample name passed to --only")
    return key


def map_generated_audio(audio_dir):
    audio_by_suffix = {}
    for path in sorted(audio_dir.glob("EN_*.es.wav")):
        name = path.name[: -len(".es.wav")]
        if not name.startswith("EN_"):
            continue
        suffix = name[3:]
        if suffix in audio_by_suffix:
            raise ValueError(f"Duplicate Spanish WAV for sample {suffix}")
        audio_by_suffix[suffix] = path
    if not audio_by_suffix:
        raise ValueError(f"No EN_*.es.wav files found in {audio_dir}")
    return audio_by_suffix


def validate_wave(path):
    import soundfile as sf

    info = sf.info(path)
    if info.frames < 1:
        raise ValueError(f"Generated WAV is empty: {path}")
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError(
            f"{path.name} must be mono {SAMPLE_RATE} Hz; "
            f"found {info.channels} channel(s) at {info.samplerate} Hz"
        )
    return info


def select_replacements(samples, audio_by_suffix, requested_keys):
    german_samples = {
        sample.name[3:]: sample
        for sample in samples
        if sample.name.startswith("DE_")
    }
    if len(german_samples) != sum(sample.name.startswith("DE_") for sample in samples):
        raise ValueError("Duplicate German sample names found in FSB bank")

    if requested_keys:
        unknown = requested_keys - german_samples.keys()
        if unknown:
            raise ValueError(
                "No matching German FSB sample for: " + ", ".join(sorted(unknown))
            )
        selected = [german_samples[key] for key in sorted(requested_keys)]
        effects = [sample for sample in selected if "_sfx_" in sample.name.lower()]
        if effects:
            raise ValueError(
                "No generated speech WAV exists for sound effects: "
                + ", ".join(sample.name for sample in effects)
            )
    else:
        selected = [
            sample
            for sample in samples
            if sample.name.startswith("DE_")
            and "_sfx_" not in sample.name.lower()
        ]

    replacement_paths = {}
    missing = []
    for sample in selected:
        suffix = sample.name[3:]
        path = audio_by_suffix.get(suffix)
        if path is None:
            missing.append(sample.name)
        else:
            replacement_paths[sample.name] = path
    if missing:
        raise FileNotFoundError(
            f"Missing Spanish WAV for {len(missing)} German sample(s): "
            + ", ".join(missing[:20])
        )
    return selected, replacement_paths, german_samples


def encode_wav_to_mp3(ffmpeg, source_wav, destination_mp3):
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_wav),
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "320k",
        "-write_xing",
        "0",
        "-write_id3v2",
        "0",
        "-write_id3v1",
        "0",
        str(destination_mp3),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"ffmpeg failed for {source_wav.name}: {result.stderr.strip()}"
        )
    if not destination_mp3.is_file() or destination_mp3.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no MP3 for {source_wav.name}")


def patch_sample_header(header_bytes, data_offset, sample_count):
    if len(header_bytes) != 8:
        raise ValueError("Expected an 8-byte FSB5 sample header")
    if data_offset % 16:
        raise ValueError(f"FSB sample offset is not 16-byte aligned: {data_offset}")
    if data_offset // 16 >= 1 << 28:
        raise ValueError("FSB5 sample data offset exceeds the format limit")
    if sample_count < 1 or sample_count >= 1 << 30:
        raise ValueError("FSB5 sample count exceeds the format limit")

    descriptor = struct.unpack("<Q", header_bytes)[0]
    if descriptor & 1:
        raise ValueError("FSB sample metadata chunks are not supported")
    descriptor &= ~(((1 << 28) - 1) << 6)
    descriptor &= ~(((1 << 30) - 1) << 34)
    descriptor |= (data_offset // 16) << 6
    descriptor |= sample_count << 34
    return struct.pack("<Q", descriptor)


def validate_ui_text_offsets(data):
    if len(data) < 4:
        raise ValueError("Truncated Spanish UI text member")
    count = struct.unpack_from("<I", data)[0]
    table_end = 4 + count * 8
    if count == 0 or table_end > len(data):
        raise ValueError("Invalid Spanish UI text offset table")
    offsets = []
    for index in range(count * 2):
        offset = struct.unpack_from("<I", data, 4 + index * 4)[0]
        target = offset + 4
        if target >= len(data) or data.find(b"\0", target) < 0:
            raise ValueError(f"Invalid Spanish UI text string offset: {offset}")
        offsets.append(target)
    return count, offsets


def patch_spanish_audio_label(data):
    count, offsets = validate_ui_text_offsets(data)
    old_value = "Alemán".encode("cp1252")
    new_value = "Español".encode("cp1252")
    old_pair = b"MENU_GERMAN\0" + old_value + b"\0"
    if data.count(old_pair) != 1:
        raise ValueError("Expected exactly one MENU_GERMAN/Alemán label")

    pair_start = data.index(old_pair)
    value_start = pair_start + len(b"MENU_GERMAN\0")
    insertion_at = value_start + len(old_value) + 1
    patched = bytearray(
        data[:value_start] + new_value + b"\0" + data[insertion_at:]
    )
    for index, target in enumerate(offsets):
        if target >= insertion_at:
            field_offset = 4 + index * 4
            offset = struct.unpack_from("<I", patched, field_offset)[0]
            if offset == 0xFFFFFFFF:
                raise ValueError("Spanish UI text offset exceeds the format limit")
            struct.pack_into("<I", patched, field_offset, offset + 1)

    validate_ui_text_offsets(patched)
    if patched.count(b"MENU_GERMAN\0Espa\xf1ol\0") != 1:
        raise ValueError("Spanish audio label patch did not validate")
    return bytes(patched)


def write_fsb_replacement(
    source_view,
    bank,
    replacement_paths,
    wave_info,
    ffmpeg,
    output_path,
):
    header = bank.header
    if header.id != b"FSB5" or header.version != 1 or header.size != 60:
        raise ValueError("Only version-1 FSB5 banks are supported")
    if header.mode != fsb5.SoundFormat.MPEG:
        raise ValueError("Only MPEG/MP3 FSB5 banks are supported")
    if header.sampleHeadersSize != header.numSamples * 8:
        raise ValueError(
            "FSB sample metadata is present; this bank layout is not supported"
        )

    source_view.seek(0)
    raw_header = bytearray(source_view.read(header.size))
    sample_headers = source_view.read(header.sampleHeadersSize)
    names_table = source_view.read(header.nameTableSize)
    if len(raw_header) != header.size:
        raise ValueError("Truncated FSB5 header")
    if len(sample_headers) != header.sampleHeadersSize:
        raise ValueError("Truncated FSB5 sample headers")
    if len(names_table) != header.nameTableSize:
        raise ValueError("Truncated FSB5 name table")
    if (header.size + header.sampleHeadersSize + header.nameTableSize) % FSB_ALIGNMENT:
        raise ValueError("FSB5 sample data does not begin on a 32-byte boundary")

    old_data_end = 0
    for index, sample in enumerate(bank.samples):
        if sample.dataOffset < old_data_end or sample.dataOffset % FSB_ALIGNMENT:
            raise ValueError(f"Unexpected FSB sample offset for {sample.name}")
        expected_end = (
            bank.samples[index + 1].dataOffset
            if index + 1 < len(bank.samples)
            else header.dataSize
        )
        if expected_end < sample.dataOffset:
            raise ValueError(f"Invalid FSB sample extent for {sample.name}")
        old_data_end = expected_end
    if old_data_end != header.dataSize or header.dataSize % FSB_ALIGNMENT:
        raise ValueError("FSB5 sample data area has an unexpected size or alignment")

    patched_headers = []
    report_rows = []
    with output_path.open("xb") as output:
        output.write(raw_header)
        output.write(sample_headers)
        output.write(names_table)
        data_size = 0
        previous_sample_offset = -1
        replacement_index = 0
        mp3_path = output_path.with_name("replacement-sample.mp3")

        try:
            for index, sample in enumerate(bank.samples):
                if sample.dataOffset < previous_sample_offset:
                    raise ValueError("FSB samples are not ordered by data offset")
                previous_sample_offset = sample.dataOffset
                new_offset = data_size
                source_header = sample_headers[index * 8 : index * 8 + 8]

                generated_wav = replacement_paths.get(sample.name)
                if generated_wav is None:
                    payload = bank._rebuild_sample(sample)
                    expected_size = original_sample_size(
                        bank.samples, header.dataSize, index
                    )
                    if len(payload) != expected_size:
                        raise ValueError(
                            f"Could not read complete source sample: {sample.name}"
                        )
                    sample_count = sample.samples
                else:
                    replacement_index += 1
                    print(
                        f"[{replacement_index}/{len(replacement_paths)}] "
                        f"{sample.name}",
                        flush=True,
                    )
                    if sample.frequency != SAMPLE_RATE or sample.channels != 1:
                        raise ValueError(
                            f"{sample.name} is not mono {SAMPLE_RATE} Hz in the source bank"
                        )
                    if sample.samples < 1:
                        raise ValueError(f"Invalid original sample length: {sample.name}")
                    encode_wav_to_mp3(ffmpeg, generated_wav, mp3_path)
                    payload = mp3_path.read_bytes()
                    sample_count = wave_info[sample.name].frames
                    old_seconds = sample.samples / sample.frequency
                    new_seconds = sample_count / SAMPLE_RATE
                    report_rows.append(
                        {
                            "sample": sample.name,
                            "wav": generated_wav.name,
                            "source_seconds": f"{old_seconds:.6f}",
                            "generated_seconds": f"{new_seconds:.6f}",
                            "duration_ratio": f"{new_seconds / old_seconds:.6f}",
                            "mp3_bytes": len(payload),
                        }
                    )

                patched_headers.append(
                    patch_sample_header(source_header, new_offset, sample_count)
                )
                output.write(payload)
                padded_size = align_up(len(payload), FSB_ALIGNMENT)
                padding = padded_size - len(payload)
                if padding:
                    output.write(b"\0" * padding)
                data_size += padded_size

            if data_size >= 1 << 32:
                raise ValueError("FSB5 data area exceeds the format limit")
            struct.pack_into("<I", raw_header, 20, data_size)
            output.seek(0)
            output.write(raw_header)
            output.seek(header.size)
            output.write(b"".join(patched_headers))
            output.flush()
            os.fsync(output.fileno())
        finally:
            if mp3_path.exists():
                mp3_path.unlink()

    expected_size = header.size + header.sampleHeadersSize + header.nameTableSize + data_size
    if output_path.stat().st_size != expected_size:
        raise ValueError("Rebuilt FSB5 size does not match its header")

    with output_path.open("rb") as rebuilt_stream:
        rebuilt = fsb5.FSB5(rebuilt_stream)
        if len(rebuilt.samples) != len(bank.samples):
            raise ValueError("Rebuilt FSB5 has a different sample count")
        for sample_name in replacement_paths:
            old_sample = bank.sampleIndex[sample_name]
            new_sample = rebuilt.sampleIndex[sample_name]
            if new_sample.samples != wave_info[sample_name].frames:
                raise ValueError(f"Rebuilt sample length mismatch: {sample_name}")
            if new_sample.frequency != old_sample.frequency:
                raise ValueError(f"Rebuilt sample rate mismatch: {sample_name}")
    return report_rows


def plan_lpak_repack(entries, replacement_sizes, data_offset, archive_size):
    by_name = {entry.name: entry for entry in entries}
    missing = set(replacement_sizes) - by_name.keys()
    if missing:
        raise ValueError("LPAK member not found: " + ", ".join(sorted(missing)))

    new_offsets = {}
    old_cursor = 0
    new_cursor = 0
    for entry in sorted(entries, key=lambda item: item.relative_offset):
        gap = entry.relative_offset - old_cursor
        if gap < 0:
            raise ValueError(f"Overlapping LPAK member: {entry.name}")
        new_cursor += gap
        new_offsets[entry.name] = new_cursor
        new_size = replacement_sizes.get(entry.name, entry.stored_size)
        if new_offsets[entry.name] >= 1 << 32 or new_size >= 1 << 32:
            raise ValueError(f"LPAK member exceeds the format limit: {entry.name}")
        new_cursor += new_size
        old_cursor = entry.relative_offset + entry.stored_size

    trailing_size = archive_size - data_offset - old_cursor
    if trailing_size < 0:
        raise ValueError("LPAK data extends beyond the archive")
    return new_offsets, trailing_size


def write_lpak_replacements(
    source_path,
    output_path,
    entries,
    entries_offset,
    data_offset,
    replacement_paths,
):
    replacement_sizes = {
        name: path.stat().st_size for name, path in replacement_paths.items()
    }
    source_size = source_path.stat().st_size
    new_offsets, trailing_size = plan_lpak_repack(
        entries,
        replacement_sizes,
        data_offset,
        source_size,
    )
    sorted_entries = sorted(entries, key=lambda item: item.relative_offset)
    old_sizes = {entry.name: entry.stored_size for entry in entries}

    with source_path.open("rb") as source, output_path.open("xb") as output:
        copy_range(source, output, 0, data_offset)
        for entry in entries:
            raw = bytearray(entry.raw)
            new_size = replacement_sizes.get(entry.name, entry.stored_size)
            struct.pack_into("<I", raw, 0, new_offsets[entry.name])
            struct.pack_into("<I", raw, 8, new_size)
            struct.pack_into("<I", raw, 12, new_size)
            output.seek(entries_offset + entry.index * 20)
            output.write(raw)

        output.seek(data_offset)
        old_cursor = 0
        for entry in sorted_entries:
            gap = entry.relative_offset - old_cursor
            copy_range(source, output, data_offset + old_cursor, gap)
            old_cursor += gap
            if output.tell() - data_offset != new_offsets[entry.name]:
                raise ValueError(
                    f"Unexpected rebuilt LPAK offset for {entry.name}"
                )
            replacement_path = replacement_paths.get(entry.name)
            if replacement_path is not None:
                with replacement_path.open("rb") as replacement_file:
                    copy_range(
                        replacement_file,
                        output,
                        0,
                        replacement_sizes[entry.name],
                    )
            else:
                copy_range(
                    source,
                    output,
                    data_offset + entry.relative_offset,
                    entry.stored_size,
                )
            old_cursor = entry.relative_offset + entry.stored_size

        copy_range(
            source,
            output,
            data_offset + old_cursor,
            trailing_size,
        )
        output.flush()
        os.fsync(output.fileno())

    expected_size = (
        source_size
        - sum(old_sizes[name] for name in replacement_sizes)
        + sum(replacement_sizes.values())
    )
    if output_path.stat().st_size != expected_size:
        raise ValueError("Rebuilt LPAK size does not match its entries")


def validate_output_archive(
    path,
    expected_sample_names,
    replacement_paths,
    wave_info,
    expected_ui_size,
):
    with path.open("rb") as archive:
        _, _, data_offset, entries = read_lpak_entries(archive)
        fsb_matches = [entry for entry in entries if entry.name == FSB_MEMBER]
        if len(fsb_matches) != 1:
            raise ValueError(f"Rebuilt archive does not contain one {FSB_MEMBER}")
        member = fsb_matches[0]
        bank = fsb5.FSB5(
            extract_tentacle.FileView(
                archive,
                data_offset + member.relative_offset,
                member.stored_size,
            )
        )
        actual_names = [sample.name for sample in bank.samples]
        if actual_names != expected_sample_names:
            raise ValueError("Rebuilt archive changed FSB sample names or order")
        for sample_name in replacement_paths:
            if bank.sampleIndex[sample_name].samples != wave_info[sample_name].frames:
                raise ValueError(f"Rebuilt archive has wrong sample length: {sample_name}")
        ui_matches = [entry for entry in entries if entry.name == UITEXT_MEMBER]
        if len(ui_matches) != 1 or ui_matches[0].stored_size != expected_ui_size:
            raise ValueError(f"Rebuilt archive has an invalid {UITEXT_MEMBER} member")
        ui_entry = ui_matches[0]
        ui_view = extract_tentacle.FileView(
            archive,
            data_offset + ui_entry.relative_offset,
            ui_entry.stored_size,
        )
        ui_data = ui_view.read()
        validate_ui_text_offsets(ui_data)
        if ui_data.count(b"MENU_GERMAN\0Espa\xf1ol\0") != 1:
            raise ValueError("Rebuilt archive does not show 'Español' for audio German")
        if ui_data.count(b"MENU_GERMAN\0Alem\xe1n\0"):
            raise ValueError("Rebuilt archive still contains the old German audio label")


def estimate_required_space(archive_size, old_fsb_size, bank, replacement_paths, wave_info):
    estimated_data = 0
    for index, sample in enumerate(bank.samples):
        if sample.name in replacement_paths:
            frames = wave_info[sample.name].frames
            duration = frames / SAMPLE_RATE
            estimate = math.ceil(duration * BITRATE / 8) + 4096
        else:
            estimate = original_sample_size(bank.samples, bank.header.dataSize, index)
        estimated_data += align_up(estimate, FSB_ALIGNMENT)
    new_fsb_size = (
        bank.header.size
        + bank.header.sampleHeadersSize
        + bank.header.nameTableSize
        + estimated_data
    )
    new_archive_size = archive_size - old_fsb_size + new_fsb_size
    return new_fsb_size + new_archive_size + 128 * 1024 * 1024


def write_report(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "sample",
                "wav",
                "source_seconds",
                "generated_seconds",
                "duration_ratio",
                "mp3_bytes",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--only",
        action="append",
        metavar="SAMPLE",
        help="replace only this German sample; may be repeated; accepts DE_ or EN_ names",
    )
    parser.add_argument(
        "--warn-below-ratio",
        type=float,
        default=0.5,
        help="warn when generated duration is below this fraction of the original",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.warn_below_ratio:
        parser.error("--warn-below-ratio must be positive")
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")
    if not args.audio_dir.is_dir():
        parser.error(f"Audio directory not found: {args.audio_dir}")
    if args.archive.resolve() == args.output.resolve():
        parser.error("Output must not overwrite the original archive")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        parser.error(f"ffmpeg executable not found: {args.ffmpeg}")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report.resolve()
        if args.report
        else output_path.with_name(output_path.name + ".report.csv")
    )
    partial_path = output_path.with_name(output_path.name + ".partial")
    report_partial = report_path.with_name(report_path.name + ".partial")
    if report_path in (output_path, args.archive.resolve()):
        parser.error("Report path must be distinct from archive and output paths")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {report_path}")
    if partial_path.exists() or report_partial.exists():
        raise FileExistsError("A partial output already exists; remove it before retrying")

    audio_by_suffix = map_generated_audio(args.audio_dir)
    requested_keys = (
        {normalize_sample_key(value) for value in args.only}
        if args.only
        else set()
    )
    imported_soundfile = __import__("soundfile")

    with args.archive.open("rb") as archive:
        _, entries_offset, data_offset, entries = read_lpak_entries(archive)
        fsb_entries = [entry for entry in entries if entry.name == FSB_MEMBER]
        if len(fsb_entries) != 1:
            raise ValueError(f"Expected one {FSB_MEMBER} member in the archive")
        ui_entries = [entry for entry in entries if entry.name == UITEXT_MEMBER]
        if len(ui_entries) != 1:
            raise ValueError(f"Expected one {UITEXT_MEMBER} member in the archive")
        fsb_entry = fsb_entries[0]
        fsb_view = extract_tentacle.FileView(
            archive,
            data_offset + fsb_entry.relative_offset,
            fsb_entry.stored_size,
        )
        bank = fsb5.FSB5(fsb_view)
        expected_sample_names = [sample.name for sample in bank.samples]
        selected, replacement_paths, german_samples = select_replacements(
            bank.samples, audio_by_suffix, requested_keys
        )
        if not replacement_paths:
            raise ValueError("No German voice samples were selected")
        ui_entry = ui_entries[0]
        ui_view = extract_tentacle.FileView(
            archive,
            data_offset + ui_entry.relative_offset,
            ui_entry.stored_size,
        )
        patched_ui_data = patch_spanish_audio_label(ui_view.read())

        wave_info = {
            sample.name: validate_wave(replacement_paths[sample.name])
            for sample in selected
        }
        if any(sample.frequency != SAMPLE_RATE or sample.channels != 1 for sample in selected):
            raise ValueError("Selected German FSB samples are not mono 48 kHz")

        free_space = shutil.disk_usage(output_path.parent).free
        required_space = estimate_required_space(
            args.archive.stat().st_size,
            fsb_entry.stored_size,
            bank,
            replacement_paths,
            wave_info,
        )
        if free_space < required_space:
            raise OSError(
                f"Insufficient free space: approximately {required_space / (1024**3):.2f} GiB "
                f"required, {free_space / (1024**3):.2f} GiB available"
            )

        print(
            f"Replacing {len(replacement_paths)} German speech sample(s); "
            f"leaving {sum('_sfx_' in s.name.lower() for s in bank.samples if s.name.startswith('DE_'))} "
            "German sound effects unchanged."
        )
        if not requested_keys:
            unused_wavs = sorted(
                set(audio_by_suffix)
                - set(german_samples)
            )
            if unused_wavs:
                print(
                    f"Ignoring {len(unused_wavs)} Spanish WAV(s) without a German FSB sample."
                )

        try:
            with tempfile.TemporaryDirectory(
                prefix="tentacle-pack-",
                dir=output_path.parent,
            ) as temporary_directory:
                temporary_directory = Path(temporary_directory)
                fsb_path = temporary_directory / "iMUSEClient_VO.fsb"
                uitext_path = temporary_directory / "sp.uitext.info"
                with uitext_path.open("xb") as output:
                    output.write(patched_ui_data)
                report_rows = write_fsb_replacement(
                    fsb_view,
                    bank,
                    replacement_paths,
                    wave_info,
                    ffmpeg,
                    fsb_path,
                )
                low_ratios = [
                    row
                    for row in report_rows
                    if float(row["duration_ratio"]) < args.warn_below_ratio
                ]
                if low_ratios:
                    print(
                        f"WARNING: {len(low_ratios)} replacement(s) are shorter than "
                        f"{args.warn_below_ratio:.2f}x their German sample duration; "
                        "review the report and dialogue completeness."
                    )
                write_lpak_replacements(
                    args.archive,
                    partial_path,
                    entries,
                    entries_offset,
                    data_offset,
                    {
                        FSB_MEMBER: fsb_path,
                        UITEXT_MEMBER: uitext_path,
                    },
                )
                validate_output_archive(
                    partial_path,
                    expected_sample_names,
                    replacement_paths,
                    wave_info,
                    len(patched_ui_data),
                )
                report_rows.sort(key=lambda row: row["sample"])
                write_report(report_partial, report_rows)
                os.replace(report_partial, report_path)
                os.replace(partial_path, output_path)
        finally:
            if partial_path.exists():
                partial_path.unlink()
            if report_partial.exists():
                report_partial.unlink()

    print(f"Created modified archive: {output_path}")
    print(f"Replacement manifest: {report_path}")


if __name__ == "__main__":
    main()
