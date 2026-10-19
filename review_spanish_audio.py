#!/usr/bin/env python3
"""Check generated Spanish audio durations and optionally compare ASR with subtitles."""

import argparse
import csv
import os
import re
import shutil
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_DIR = ROOT / "audio"
DEFAULT_GENERATED_DIR = ROOT / "audio_es"
DEFAULT_TRANSLATIONS = ROOT / "translations_es.csv"
DEFAULT_REPORT = ROOT / "audio_es_review.csv"
DEFAULT_REVIEW_DIR = ROOT / "audio_es_review_cases"
MODEL_CACHE = ROOT / "model_cache"


def load_translations(path):
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"audio", "spanish"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Translation CSV needs audio and spanish columns")
        for row in reader:
            stem = (row.get("audio") or "").strip()
            spanish = (row.get("spanish") or "").strip()
            if not spanish:
                continue
            if not stem or stem in seen:
                raise ValueError(f"Empty or duplicate audio key in translations CSV: {stem!r}")
            seen.add(stem)
            rows.append((stem, spanish))
    if not rows:
        raise ValueError(f"No translated dialogue rows found in {path}")
    return rows


def audio_duration(path):
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as error:
        return None, str(error)
    if info.frames < 1 or info.samplerate < 1:
        return None, f"Audio has invalid dimensions: {info.frames} frames at {info.samplerate} Hz"
    return info.frames / info.samplerate, ""


def normalize_words(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def word_error_rate(reference, hypothesis):
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(
            hypothesis_words, start=1
        ):
            substitution = previous[hypothesis_index - 1] + (
                reference_word != hypothesis_word
            )
            insertion = current[hypothesis_index - 1] + 1
            deletion = previous[hypothesis_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference_words)


def build_asr(args, parser):
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE / "hub"))

    import torch
    from transformers import pipeline

    if args.asr_device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif args.asr_device == "cpu":
        device = -1
    elif args.asr_device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested for ASR but is not available")
        device = 0
    elif args.asr_device.startswith("cuda:"):
        try:
            device = int(args.asr_device.split(":", 1)[1])
        except ValueError:
            parser.error("--asr-device must be auto, cpu, cuda or cuda:N")
        if device < 0:
            parser.error("CUDA device index must not be negative")
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            parser.error(f"ASR device {args.asr_device} is not available")
    else:
        parser.error("--asr-device must be auto, cpu, cuda or cuda:N")

    return pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=device,
    )


def transcribe(transcriber, path):
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    result = transcriber(
        {"array": mono, "sampling_rate": sample_rate},
        generate_kwargs={"language": "spanish", "task": "transcribe"},
    )
    return result["text"].strip()


def review_rows(args, translations, transcriber):
    reviewed = []
    for index, (stem, spanish) in enumerate(translations, start=1):
        original_path = args.audio_dir / f"{stem}.mp3"
        generated_path = args.generated_dir / f"{stem}.es.wav"
        english_duration = None
        generated_duration = None
        errors = []

        if not original_path.is_file():
            errors.append(f"Missing original audio: {original_path}")
        else:
            english_duration, error = audio_duration(original_path)
            if error:
                errors.append(f"Invalid original audio: {error}")

        if not generated_path.is_file():
            errors.append(f"Missing generated audio: {generated_path}")
        else:
            generated_duration, error = audio_duration(generated_path)
            if error:
                errors.append(f"Invalid generated audio: {error}")

        duration_ratio = None
        duration_flag = "not_comparable"
        if english_duration and generated_duration:
            duration_ratio = generated_duration / english_duration
            if duration_ratio < args.min_duration_ratio:
                duration_flag = "short"
            elif duration_ratio > args.max_duration_ratio:
                duration_flag = "long"
            else:
                duration_flag = "ok"

        asr_text = ""
        wer = None
        if transcriber is None:
            asr_flag = "not_run"
        elif generated_duration is None:
            asr_flag = "unavailable"
        elif not args.asr_all and duration_flag not in {"short", "long"}:
            asr_flag = "not_selected"
        else:
            asr_text = transcribe(transcriber, generated_path)
            if not normalize_words(asr_text):
                wer = 1.0
                asr_flag = "empty_transcription"
            else:
                wer = word_error_rate(spanish, asr_text)
                asr_flag = (
                    "high_wer"
                    if wer > args.max_wer
                    else "ok"
                )

        if not original_path.is_file() and not generated_path.is_file():
            status = "missing_both"
        elif not original_path.is_file():
            status = "missing_original"
        elif not generated_path.is_file():
            status = "missing_generated"
        elif errors:
            status = "invalid_audio"
        else:
            status = "ok"

        reviewed.append(
            {
                "audio": stem,
                "status": status,
                "original_seconds": (
                    f"{english_duration:.3f}" if english_duration is not None else ""
                ),
                "generated_seconds": (
                    f"{generated_duration:.3f}"
                    if generated_duration is not None
                    else ""
                ),
                "duration_ratio": (
                    f"{duration_ratio:.4f}" if duration_ratio is not None else ""
                ),
                "duration_flag": duration_flag,
                "spanish_subtitle": spanish,
                "asr_text": asr_text,
                "word_error_rate": f"{wer:.4f}" if wer is not None else "",
                "asr_flag": asr_flag,
                "error": "; ".join(errors),
            }
        )
        if index % 100 == 0 or index == len(translations):
            print(f"Reviewed {index}/{len(translations)} audio files")
    return reviewed


def write_report(path, rows, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "audio",
                "status",
                "original_seconds",
                "generated_seconds",
                "duration_ratio",
                "duration_flag",
                "spanish_subtitle",
                "asr_text",
                "word_error_rate",
                "asr_flag",
                "error",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def is_potential_problem(row):
    return (
        row["status"] != "ok"
        or row["duration_flag"] in {"short", "long"}
        or row["asr_flag"] in {"high_wer", "empty_transcription", "unavailable"}
    )


def copy_review_assets(args, rows, selected_stems):
    review_dir = args.review_dir.resolve()
    source_dirs = (args.audio_dir.resolve(), args.generated_dir.resolve())
    if any(
        review_dir == source_dir or review_dir.is_relative_to(source_dir)
        for source_dir in source_dirs
    ):
        raise ValueError("Review directory must not be inside an audio source directory")

    problem_stems = {row["audio"] for row in rows}
    files_by_stem = {}
    missing = []
    existing_destinations = []
    for row in rows:
        stem = row["audio"]
        sources = (
            args.audio_dir / f"{stem}.mp3",
            args.audio_dir / f"{stem}.txt",
            args.generated_dir / f"{stem}.es.wav",
            args.audio_dir / f"{stem}.es.txt",
        )
        for source in sources:
            if not source.is_file():
                missing.append(str(source))
        files_by_stem[stem] = sources
        existing_destinations.extend(
            review_dir / stem / source.name
            for source in sources
            if (review_dir / stem / source.name).exists()
        )

    if missing:
        raise FileNotFoundError(
            "Cannot create complete review bundles; missing required files:\n"
            + "\n".join(missing)
        )
    if existing_destinations and not args.overwrite:
        raise FileExistsError(
            "Review bundle files already exist; use --overwrite to replace them"
        )

    stale_count = 0
    if review_dir.is_dir():
        for stem in selected_stems - problem_stems:
            stale_dir = review_dir / stem
            if not stale_dir.is_dir():
                continue
            entries = list(stale_dir.iterdir())
            expected_names = {
                f"{stem}.mp3",
                f"{stem}.txt",
                f"{stem}.es.wav",
                f"{stem}.es.txt",
            }
            if entries and all(
                entry.is_file() and entry.name in expected_names
                for entry in entries
            ):
                for entry in entries:
                    entry.unlink()
                stale_dir.rmdir()
                stale_count += 1
    if stale_count:
        print(f"Removed {stale_count} resolved review folders")

    if not rows:
        return 0

    for stem, sources in files_by_stem.items():
        destination_dir = review_dir / stem
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            shutil.copy2(source, destination_dir / source.name)
    return len(files_by_stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_REVIEW_DIR,
        help="directory for per-audio folders containing audios and subtitle text",
    )
    parser.add_argument(
        "--min-duration-ratio",
        type=float,
        default=0.5,
        help="flag generated audio below this fraction of the English original",
    )
    parser.add_argument(
        "--max-duration-ratio",
        type=float,
        default=2.0,
        help="flag generated audio above this multiple of the English original",
    )
    parser.add_argument(
        "--asr",
        action="store_true",
        help="transcribe duration-anomalous WAVs and compare with Spanish subtitles",
    )
    parser.add_argument(
        "--asr-all",
        action="store_true",
        help="with --asr, transcribe every selected WAV instead of duration anomalies only",
    )
    parser.add_argument(
        "--report-all",
        action="store_true",
        help="include non-problem files in the CSV report",
    )
    parser.add_argument("--asr-model", default="openai/whisper-small")
    parser.add_argument(
        "--asr-device",
        default="auto",
        help="auto, cpu, cuda or cuda:N",
    )
    parser.add_argument(
        "--max-wer",
        type=float,
        default=0.5,
        help="flag ASR/subtitle word error rates above this value",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="review only this audio stem; may be repeated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="review at most this many translated dialogues",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report and matching review-bundle files",
    )
    args = parser.parse_args()

    if args.min_duration_ratio < 0:
        parser.error("--min-duration-ratio must not be negative")
    if args.max_duration_ratio <= args.min_duration_ratio:
        parser.error("--max-duration-ratio must exceed --min-duration-ratio")
    if not 0 <= args.max_wer <= 1:
        parser.error("--max-wer must be between 0 and 1")
    if args.asr_all and not args.asr:
        parser.error("--asr-all requires --asr")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.audio_dir.is_dir():
        parser.error(f"Audio directory not found: {args.audio_dir}")
    if not args.generated_dir.is_dir():
        parser.error(f"Generated audio directory not found: {args.generated_dir}")
    if not args.translations.is_file():
        parser.error(f"Translations CSV not found: {args.translations}")
    if args.report.resolve() == args.translations.resolve():
        parser.error("Report must not overwrite the translations CSV")
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {args.report}; use --overwrite")

    translations = load_translations(args.translations)
    if args.only:
        requested = set(args.only)
        available = {stem for stem, _ in translations}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested audio stems are not translated: {', '.join(sorted(missing))}"
            )
        translations = [row for row in translations if row[0] in requested]
    if args.limit is not None:
        translations = translations[: args.limit]
    if not translations:
        raise ValueError("No translated audio selected for review")

    transcriber = build_asr(args, parser) if args.asr else None
    rows = review_rows(args, translations, transcriber)
    problem_rows = [row for row in rows if is_potential_problem(row)]
    report_rows = rows if args.report_all else problem_rows
    selected_stems = {stem for stem, _ in translations}
    copied_cases = copy_review_assets(args, problem_rows, selected_stems)
    write_report(args.report, report_rows, args.overwrite)

    short_count = sum(row["duration_flag"] == "short" for row in rows)
    long_count = sum(row["duration_flag"] == "long" for row in rows)
    missing_count = sum(row["status"].startswith("missing") for row in rows)
    invalid_count = sum(row["status"] == "invalid_audio" for row in rows)
    asr_count = sum(
        row["asr_flag"] in {"ok", "high_wer", "empty_transcription"}
        for row in rows
    )
    high_wer_count = sum(row["asr_flag"] == "high_wer" for row in rows)
    empty_asr_count = sum(row["asr_flag"] == "empty_transcription" for row in rows)
    print(
        f"Reviewed {len(rows)} files: {short_count} short, {long_count} long, "
        f"{missing_count} missing, {invalid_count} invalid; "
        f"ASR analyzed {asr_count}, {high_wer_count} high-WER, "
        f"{empty_asr_count} empty ASR transcripts. "
        f"Report: {args.report} ({len(report_rows)} rows); "
        f"review bundles: {copied_cases} in {args.review_dir}"
    )


if __name__ == "__main__":
    main()
