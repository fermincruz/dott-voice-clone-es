#!/usr/bin/env python3
"""Build one longest-clip voice reference per character, joining short clips."""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
DEFAULT_OUTPUT = ROOT / "voice_references"
MODEL_CACHE = ROOT / "model_cache"
GAP_SECONDS = 0.25


def character_code(stem):
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "EN":
        return None
    code = parts[1]
    if code.lower() == "sfx":
        return None
    if not re.fullmatch(r"[A-Z0-9]{3}", code):
        raise ValueError(f"Cannot identify character code in audio filename: {stem}")
    return code


def select_clips(audio_dir, minimum_seconds):
    import soundfile as sf

    groups = defaultdict(list)
    for path in sorted(audio_dir.glob("EN_*.mp3")):
        code = character_code(path.stem)
        if code is None:
            continue
        duration = sf.info(path).duration
        groups[code].append((duration, path))

    if not groups:
        raise ValueError(f"No character dialogue MP3s found in {audio_dir}")

    selections = {}
    for code, clips in sorted(groups.items()):
        clips.sort(key=lambda item: (-item[0], item[1].stem))
        selected = []
        speech_seconds = 0.0
        for duration, path in clips:
            selected.append((duration, path))
            speech_seconds += duration
            if speech_seconds >= minimum_seconds:
                break
        selections[code] = (selected, speech_seconds)
    return selections


def build_reference(selected, gap_seconds):
    import numpy as np
    import soundfile as sf

    parts = []
    sample_rate = None
    for index, (_, path) in enumerate(selected):
        audio, current_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch while joining reference clips: {path}"
            )
        mono = audio.mean(axis=1)
        parts.append(mono)
        if index + 1 < len(selected) and gap_seconds:
            parts.append(np.zeros(round(gap_seconds * sample_rate), dtype=np.float32))
    return np.concatenate(parts), sample_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=5.0,
        help="join longest clips until this much speech is available",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=GAP_SECONDS,
        help="silence inserted between joined clips",
    )
    parser.add_argument("--model-id", default="openai/whisper-base.en")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_seconds <= 0:
        parser.error("--min-seconds must be positive")
    if args.gap_seconds < 0:
        parser.error("--gap-seconds must not be negative")
    if not args.audio_dir.is_dir():
        parser.error(f"Audio directory not found: {args.audio_dir}")

    selections = select_clips(args.audio_dir, args.min_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    output_files = []
    for code in selections:
        output_files.extend(
            (args.output / f"{code}.wav", args.output / f"{code}.txt")
        )
    output_files.append(args.output / "voice_references.csv")
    if not args.overwrite:
        existing = [path for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} voice reference files already exist; use --overwrite"
            )

    if args.device == "cpu":
        asr_device = -1
    elif args.device == "cuda":
        asr_device = 0
    elif args.device.startswith("cuda:"):
        asr_device = int(args.device.split(":", 1)[1])
    else:
        parser.error("--device must be cpu, cuda or cuda:N")

    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE / "hub"))
    from transformers import pipeline
    import soundfile as sf

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        device=asr_device,
    )
    manifest = []
    for index, (code, (selected, speech_seconds)) in enumerate(
        selections.items(), start=1
    ):
        wave, sample_rate = build_reference(selected, args.gap_seconds)
        result = transcriber(
            {"array": wave, "sampling_rate": sample_rate}
        )["text"].strip()
        if not result:
            raise ValueError(f"ASR returned an empty transcript for character {code}")

        audio_path = args.output / f"{code}.wav"
        text_path = args.output / f"{code}.txt"
        sf.write(audio_path, wave, sample_rate)
        text_path.write_text(result + "\n", encoding="utf-8")
        manifest.append(
            {
                "character": code,
                "reference_audio": audio_path.name,
                "prompt_text": text_path.name,
                "source_clips": ";".join(path.stem for _, path in selected),
                "source_clip_count": len(selected),
                "speech_seconds": f"{speech_seconds:.3f}",
                "reference_seconds": f"{len(wave) / sample_rate:.3f}",
                "meets_minimum": str(speech_seconds >= args.min_seconds).lower(),
                "asr_model": args.model_id,
                "transcript": result,
            }
        )
        print(
            f"[{index}/{len(selections)}] {code}: "
            f"{len(selected)} clip(s), {speech_seconds:.2f}s speech"
        )

    manifest_path = args.output / "voice_references.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    short = [row["character"] for row in manifest if row["meets_minimum"] == "false"]
    print(
        f"Created {len(manifest)} character references in {args.output}; "
        f"below {args.min_seconds:g}s: {', '.join(short) if short else 'none'}"
    )


if __name__ == "__main__":
    main()
