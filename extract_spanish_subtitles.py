#!/usr/bin/env python3
"""Extract Spanish subtitles from tenta.cle into .es.txt files and the TTS CSV."""

import argparse
import csv
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import extract_tentacle as extractor  # noqa: E402


NO_SPANISH_SUBTITLE = "[No Spanish subtitle in game index]"


def collect_subtitles(info, sample_name):
    return [
        value.strip()
        for value in extractor.find_subtitle_values(info, sample_name)
        if value.strip()
    ]


def select_translation(english_text, spanish_values, sample_name):
    translated_values = [
        text for text in spanish_values if text != english_text
    ]
    if len(translated_values) > 1:
        raise ValueError(f"Conflicting Spanish translations for {sample_name}")
    if translated_values:
        return translated_values[0]
    return spanish_values[0] if spanish_values else None


def write_merge_report(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("audio", "subtitle_key", "next_audio", "english", "spanish"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=BASE_DIR / "source" / "tenta.cle")
    parser.add_argument("--audio-dir", type=Path, default=BASE_DIR / "audio")
    parser.add_argument(
        "--translations",
        type=Path,
        default=BASE_DIR / "translations_es.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing Spanish translations and .es.txt files",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")
    if not args.audio_dir.is_dir():
        parser.error(f"Audio directory not found: {args.audio_dir}")

    audio_paths = sorted(args.audio_dir.glob("EN_*.mp3"))
    if not audio_paths:
        parser.error(f"No EN_*.mp3 files found in {args.audio_dir}")
    audio_by_stem = {path.stem: path for path in audio_paths}
    if len(audio_by_stem) != len(audio_paths):
        raise ValueError("Duplicate audio stems found")

    if args.translations.is_file():
        with args.translations.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != ["audio", "english", "spanish"]:
                raise ValueError("CSV must have columns: audio, english, spanish")
            rows = list(reader)
            fieldnames = reader.fieldnames
    else:
        rows = []
        fieldnames = ["audio", "english", "spanish"]

    row_by_stem = {}
    for row in rows:
        stem = row["audio"]
        if stem in row_by_stem:
            raise ValueError(f"Duplicate audio row in CSV: {stem}")
        if stem not in audio_by_stem:
            raise ValueError(f"CSV audio has no matching MP3: {stem}")
        row_by_stem[stem] = row

    with args.archive.open("rb") as archive:
        data_offset, entries = extractor.read_index(archive)
        english_info = extractor.open_member(
            archive, data_offset, entries, "localization/en.speech.info"
        ).read()
        spanish_info = extractor.open_member(
            archive, data_offset, entries, "localization/sp.speech.info"
        ).read()
        all_sample_names = extractor.read_voice_sample_names(
            archive, data_offset, entries
        )

    sample_names = list(audio_by_stem)
    subtitle_merges = extractor.find_orphan_subtitle_merges(
        english_info,
        sample_names,
        all_sample_names,
    )
    assigned_keys = extractor.assigned_subtitle_keys(
        sample_names,
        subtitle_merges,
    )

    output_texts = {}
    spanish_by_stem = {}
    english_by_stem = {}
    for stem in audio_by_stem:
        sample_name = stem
        english_parts = []
        spanish_parts = []
        for subtitle_key in assigned_keys[sample_name]:
            subtitle_sample = f"EN_{subtitle_key}"
            english_values = collect_subtitles(english_info, subtitle_sample)
            if len(english_values) > 1:
                raise ValueError(
                    f"Conflicting English subtitles for {subtitle_sample}"
                )
            english_text = english_values[0] if english_values else None
            if english_text:
                english_parts.append(english_text)

            spanish_values = collect_subtitles(spanish_info, subtitle_sample)
            spanish_text = select_translation(
                english_text,
                spanish_values,
                subtitle_sample,
            )
            if spanish_text:
                spanish_parts.append(spanish_text)

        english_by_stem[stem] = " ".join(english_parts) or None
        spanish_by_stem[stem] = " ".join(spanish_parts) or None
        output_texts[stem] = spanish_by_stem[stem] or NO_SPANISH_SUBTITLE

    dialogue_stems = {
        stem for stem, english_text in english_by_stem.items() if english_text
    }
    if not args.translations.is_file():
        rows = [
            {
                "audio": stem,
                "english": english_by_stem[stem],
                "spanish": spanish_by_stem[stem] or "",
            }
            for stem in sorted(dialogue_stems)
        ]
        row_by_stem = {row["audio"]: row for row in rows}

    if set(row_by_stem) != dialogue_stems:
        raise ValueError(
            "CSV dialogue rows do not match the English subtitles in the source archive"
        )

    distinct_count = same_count = missing_count = 0
    for stem, row in row_by_stem.items():
        english_text = english_by_stem[stem]
        if row["english"] != english_text and not args.overwrite:
            raise ValueError(f"English CSV text does not match source subtitle: {stem}")
        row["english"] = english_text
        spanish_text = spanish_by_stem[stem]
        if spanish_text is None:
            missing_count += 1
        else:
            if spanish_text == english_text:
                same_count += 1
            else:
                distinct_count += 1

    if not args.overwrite:
        for stem, row in row_by_stem.items():
            existing = row["spanish"].strip()
            expected = spanish_by_stem[stem] or ""
            if existing and existing != expected:
                raise FileExistsError(
                    f"CSV already contains a different Spanish translation for {stem}; "
                    "use --overwrite to replace it"
                )
        for stem, text in output_texts.items():
            path = args.audio_dir / f"{stem}.es.txt"
            if path.exists():
                existing = path.read_text(encoding="utf-8").rstrip("\r\n")
                if existing != text:
                    raise FileExistsError(
                        f"Spanish subtitle file already differs from source: {path}; "
                        "use --overwrite to replace it"
                    )

    for stem, row in row_by_stem.items():
        if args.overwrite or not row["spanish"].strip():
            row["spanish"] = spanish_by_stem[stem] or ""

    for stem, text in output_texts.items():
        path = args.audio_dir / f"{stem}.es.txt"
        with path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(text + "\n")

    with args.translations.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    merge_report = []
    for merge in subtitle_merges:
        subtitle_sample = f"EN_{merge['subtitle_key']}"
        english_values = collect_subtitles(english_info, subtitle_sample)
        if len(english_values) > 1:
            raise ValueError(f"Conflicting English subtitles for {subtitle_sample}")
        english_text = english_values[0] if english_values else ""
        spanish_values = collect_subtitles(spanish_info, subtitle_sample)
        spanish_text = select_translation(
            english_text or None,
            spanish_values,
            subtitle_sample,
        )
        merge_report.append(
            {
                "audio": merge["audio"],
                "subtitle_key": subtitle_sample,
                "next_audio": merge["next_audio"],
                "english": english_text,
                "spanish": spanish_text or "",
            }
        )
    merge_report_path = args.translations.with_name("subtitle_merges.csv")
    write_merge_report(merge_report_path, merge_report)

    print(f"Audio files with Spanish sidecars: {len(output_texts)}")
    print(f"Spanish dialogue entries distinct from English: {distinct_count}")
    print(f"Spanish entries identical to English: {same_count}")
    print(f"Dialogue entries without Spanish text: {missing_count}")
    print(f"Updated translations: {args.translations}")
    print(
        f"Attached {len(subtitle_merges)} subtitle-only fragment(s); "
        f"merge report: {merge_report_path}"
    )


if __name__ == "__main__":
    main()
