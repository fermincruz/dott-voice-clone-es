#!/usr/bin/env python3
"""Generate Spanish VoxCPM2 speech using clip or character voice references."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
DEFAULT_TRANSLATIONS = ROOT / "translations_es.csv"
DEFAULT_OUTPUT = ROOT / "audio_es"
MODEL_CACHE = ROOT / "model_cache"
MIN_CLIP_REFERENCE_SECONDS = 1.0


def is_subtitle(text):
    return text and not text.startswith("[")


def character_code(stem):
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "EN":
        raise ValueError(f"Cannot identify character code in audio filename: {stem}")
    code = parts[1]
    if not code.isalnum() or len(code) != 3:
        raise ValueError(f"Invalid character code in audio filename: {stem}")
    return code


def make_template(path):
    rows = []
    for english_file in sorted(AUDIO_DIR.glob("*.txt")):
        english = english_file.read_text(encoding="utf-8").strip()
        if is_subtitle(english):
            rows.append(
                {
                    "audio": english_file.stem,
                    "english": english,
                    "spanish": "",
                }
            )
    if path.exists():
        raise FileExistsError(f"Translation template already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("audio", "english", "spanish"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} untranslated dialogue rows to {path}")


def load_jobs(args):
    prompt_override = None
    if args.prompt_text_file is not None:
        if not args.only:
            raise ValueError("--prompt-text-file requires --only <audio-stem>")
        if not args.prompt_text_file.is_file():
            raise FileNotFoundError(f"Prompt transcript not found: {args.prompt_text_file}")
        prompt_override = args.prompt_text_file.read_text(encoding="utf-8-sig").strip()
        if not prompt_override:
            raise ValueError(f"Prompt transcript is empty: {args.prompt_text_file}")

    if args.test_text is not None:
        if not args.only:
            raise ValueError("--test-text requires --only <audio-stem>")
        if not args.test_text.strip():
            raise ValueError("--test-text must not be empty")
        audio = AUDIO_DIR / f"{args.only}.mp3"
        transcript = AUDIO_DIR / f"{args.only}.txt"
        if not audio.is_file() or not transcript.is_file():
            raise FileNotFoundError(f"Audio/TXT pair not found for {args.only}")
        english = transcript.read_text(encoding="utf-8").strip()
        if not is_subtitle(english):
            raise ValueError(f"{args.only} has no indexed English dialogue")
        return [(args.only, audio, prompt_override or english, args.test_text)]

    if not args.translations.is_file():
        raise FileNotFoundError(
            f"Translation CSV not found: {args.translations}. "
            "Use --make-template first, then fill its Spanish column."
        )

    jobs = []
    seen = set()
    with args.translations.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"audio", "english", "spanish"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Translation CSV needs audio, english and spanish columns")
        for row in reader:
            stem = (row.get("audio") or "").strip()
            spanish = (row.get("spanish") or "").strip()
            if not spanish:
                continue
            if not stem or stem in seen:
                raise ValueError(f"Empty or duplicate audio key in translations CSV: {stem!r}")
            seen.add(stem)
            audio = AUDIO_DIR / f"{stem}.mp3"
            transcript = AUDIO_DIR / f"{stem}.txt"
            if not audio.is_file() or not transcript.is_file():
                raise FileNotFoundError(f"Missing audio/TXT pair for {stem}")
            english = transcript.read_text(encoding="utf-8").strip()
            if not is_subtitle(english):
                raise ValueError(f"{stem} has no indexed English dialogue")
            if (row.get("english") or "").strip() != english:
                raise ValueError(f"English text in CSV does not match {transcript.name}")
            if prompt_override is not None:
                english = prompt_override
            jobs.append((stem, audio, english, spanish))
    if args.only:
        jobs = [job for job in jobs if job[0] == args.only]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("No translated dialogue rows selected")
    return jobs


def translation_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_synthesis_manifest(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("translations"), dict)
        or not all(
            isinstance(stem, str) and isinstance(text_hash, str)
            for stem, text_hash in manifest["translations"].items()
        )
    ):
        raise ValueError(f"Invalid synthesis manifest: {path}")
    return manifest["translations"]


def save_synthesis_manifest(path, translations):
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(
            {"version": 1, "translations": dict(sorted(translations.items()))},
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")
    os.replace(temporary_path, path)


def baseline_translation_hash(stem):
    baseline_path = AUDIO_DIR / f"{stem}.es.txt"
    if not baseline_path.is_file():
        return None
    baseline = baseline_path.read_text(encoding="utf-8-sig").strip()
    return translation_hash(baseline)


def select_jobs_to_synthesize(jobs, output_dir, manifest, force=False):
    selected = []
    unchanged_count = 0
    for job in jobs:
        stem, _, _, spanish = job
        output_path = output_dir / f"{stem}.es.wav"
        current_hash = translation_hash(spanish)
        last_synthesized_hash = manifest.get(stem)

        if not force and output_path.is_file():
            if last_synthesized_hash == current_hash:
                unchanged_count += 1
                continue
            if (
                last_synthesized_hash is None
                and baseline_translation_hash(stem) == current_hash
            ):
                unchanged_count += 1
                continue
        selected.append(job)
    return selected, unchanged_count


def record_synthesized_translation(stem, spanish, manifest):
    current_hash = translation_hash(spanish)
    if baseline_translation_hash(stem) == current_hash:
        manifest.pop(stem, None)
    else:
        manifest[stem] = current_hash


def resolve_reference_audio(
    jobs,
    reference_dir,
    reference_source,
    cloning_mode,
    prompt_text_file=None,
):
    import librosa

    prompt_override = None
    if cloning_mode == "prompt" and prompt_text_file is not None:
        prompt_override = prompt_text_file.read_text(
            encoding="utf-8-sig"
        ).strip()
        if not prompt_override:
            raise ValueError(f"Prompt transcript is empty: {prompt_text_file}")

    resolved_jobs = []
    fallback_stems = []
    for stem, audio, english, spanish in jobs:
        use_character_reference = reference_source == "character"
        reference_transcript = english
        if reference_source == "clip":
            duration = librosa.get_duration(path=str(audio))
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"Invalid reference audio duration for {stem}: {duration}"
                )
            use_character_reference = duration < MIN_CLIP_REFERENCE_SECONDS
            if use_character_reference:
                fallback_stems.append(stem)

        if use_character_reference:
            if reference_dir is None:
                raise ValueError(
                    f"{stem} needs a character reference; pass --voice-references"
                )
            code = character_code(stem)
            reference_audio = reference_dir / f"{code}.wav"
            if not reference_audio.is_file():
                raise FileNotFoundError(
                    f"Missing voice reference for character {code}: "
                    f"{reference_audio}. Run prepare_voice_references.py first."
                )
            if cloning_mode == "prompt":
                if prompt_override is not None:
                    reference_transcript = prompt_override
                else:
                    transcript_file = reference_dir / f"{code}.txt"
                    if not transcript_file.is_file():
                        raise FileNotFoundError(
                            f"Missing voice reference transcript: {transcript_file}"
                        )
                    reference_transcript = transcript_file.read_text(
                        encoding="utf-8-sig"
                    ).strip()
                if not reference_transcript:
                    prompt_source = (
                        prompt_text_file
                        if prompt_override is not None
                        else transcript_file
                    )
                    raise ValueError(
                        f"Prompt transcript is empty: {prompt_source}"
                    )
        else:
            reference_audio = audio

        resolved_jobs.append(
            (
                stem,
                reference_audio,
                reference_transcript,
                spanish,
                use_character_reference and reference_source == "clip",
            )
        )
    return resolved_jobs, fallback_stems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--make-template", type=Path)
    parser.add_argument("--only", help="one audio stem to run")
    parser.add_argument(
        "--test-text",
        help="temporary Spanish test sentence; requires --only and bypasses the CSV",
    )
    parser.add_argument(
        "--prompt-text-file",
        type=Path,
        help="complete transcript for the reference audio; requires --only",
    )
    parser.add_argument(
        "--voice-references",
        type=Path,
        help="per-character WAV/TXT references; required for character mode or short-clip fallback",
    )
    parser.add_argument(
        "--reference-source",
        choices=("clip", "character"),
        default="clip",
        help="use each English clip as reference, falling back to character audio below 1 second; or always use character audio",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default="openbmb/VoxCPM2")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cloning-mode",
        choices=("reference", "prompt"),
        default="reference",
        help="reference (default) omits prompt audio/text; prompt uses the legacy transcript method",
    )
    parser.add_argument(
        "--reference-only",
        dest="cloning_mode",
        action="store_const",
        const="reference",
        default="reference",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--limit", type=int, help="maximum translated clips to process")
    parser.add_argument("--optimize", action="store_true", help="enable torch.compile")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="force regeneration of all selected clips, replacing existing WAVs",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    if args.make_template:
        make_template(args.make_template)
        return
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE / "hub"))
    jobs = load_jobs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / ".translation_manifest.json"
    manifest = (
        {}
        if args.test_text is not None
        else load_synthesis_manifest(manifest_path)
    )
    unchanged_count = 0
    if args.test_text is None:
        selected_count = len(jobs)
        jobs, unchanged_count = select_jobs_to_synthesize(
            jobs,
            args.output,
            manifest,
            force=args.overwrite,
        )
        print(
            f"Selected {selected_count} translated clips: "
            f"{len(jobs)} new or changed; {unchanged_count} unchanged"
        )
        if not jobs:
            print("Nothing to synthesize; all selected WAVs match the translations.")
            return
    elif not args.overwrite:
        existing = [
            args.output / f"{stem}.es.wav"
            for stem, *_ in jobs
            if (args.output / f"{stem}.es.wav").exists()
        ]
        if existing:
            raise FileExistsError(
                f"{len(existing)} output WAV files already exist; use --overwrite"
            )

    jobs, fallback_stems = resolve_reference_audio(
        jobs,
        args.voice_references,
        args.reference_source,
        args.cloning_mode,
        args.prompt_text_file,
    )
    if args.reference_source == "clip":
        print(
            f"Using English clips as references; falling back to character "
            f"references below {MIN_CLIP_REFERENCE_SECONDS:.1f}s "
            f"for {len(fallback_stems)} of {len(jobs)} clips"
        )
    else:
        print("Using character voice references for all clips")

    from voxcpm import VoxCPM
    import soundfile as sf
    import torch

    torch.manual_seed(args.seed)
    model = VoxCPM.from_pretrained(
        args.model_id,
        device=args.device,
        optimize=args.optimize,
        load_denoiser=False,
    )
    for index, (
        stem,
        reference_audio,
        prompt_text,
        spanish,
        character_fallback,
    ) in enumerate(jobs, start=1):
        fallback_message = (
            "; clip under 1 second, using character reference"
            if character_fallback
            else ""
        )
        print(
            f"[{index}/{len(jobs)}] {stem} "
            f"(reference: {reference_audio.name}{fallback_message})"
        )
        generation_args = {
            "text": spanish,
            "reference_wav_path": str(reference_audio),
            "cfg_value": args.cfg_value,
            "inference_timesteps": args.timesteps,
        }
        if args.cloning_mode == "prompt":
            generation_args["prompt_wav_path"] = str(reference_audio)
            generation_args["prompt_text"] = prompt_text
        wav = model.generate(**generation_args)
        output_path = args.output / f"{stem}.es.wav"
        sf.write(output_path, wav, model.tts_model.sample_rate)
        if args.test_text is None:
            record_synthesized_translation(stem, spanish, manifest)
            if index % 50 == 0 or index == len(jobs):
                if manifest or manifest_path.exists():
                    save_synthesis_manifest(manifest_path, manifest)
    print(f"Generated {len(jobs)} WAV files in {args.output}")


if __name__ == "__main__":
    main()
