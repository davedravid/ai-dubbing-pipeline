"""Command-line entry point: `python -m dubbing <command> ...`."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

from dubbing.config import load_settings
from dubbing.errors import PipelineError
from dubbing.logging_setup import setup_logging
from dubbing.paths import JobPaths, validate_job_id

log = logging.getLogger("dubbing")


def _default_job_id(input_path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(input_path).stem).strip("_")
    return (stem or "job")[:64]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dubbing", description="AI Dubbing Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="Extract audio from a video and save a timestamped transcript")
    t.add_argument("input", help="Path to a video or audio file")
    t.add_argument("--job-id", help="Name for this run's files (default: based on the file name)")
    t.add_argument("--preset", choices=["fast", "quality"], help="Override the preset in config.yaml")
    t.add_argument("--language", help="Source language code, e.g. 'en' (default: auto-detect)")
    t.add_argument("--config", type=Path, help="Use a different config file")
    t.add_argument("--overwrite", action="store_true", help="Replace this job's existing transcript")
    t.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    sc = sub.add_parser("script", help="Build the dubbing script (speakers, timings, sounds, voice samples) + review page")
    sc.add_argument("--job-id", required=True)
    sc.add_argument("--config", type=Path, help="Use a different config file")
    sc.add_argument("--overwrite", action="store_true", help="Build the script again (e.g. after correcting speakers)")
    sc.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    tr = sub.add_parser("translate", help="Translate a job's transcript into the target language(s)")
    tr.add_argument("--job-id", required=True, help="The job whose transcript to translate")
    tr.add_argument("--to", help="Target language code(s), comma-separated, e.g. 'hi' (default: config.yaml)")
    tr.add_argument("--config", type=Path, help="Use a different config file")
    tr.add_argument("--overwrite", action="store_true", help="Replace an existing translation")
    tr.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    a = sub.add_parser("analyze", help="Measure pitch, loudness and pace of each original line")
    a.add_argument("--job-id", required=True)
    a.add_argument("--config", type=Path, help="Use a different config file")
    a.add_argument("--overwrite", action="store_true", help="Replace existing measurements")
    a.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    sp = sub.add_parser("speak", help="Generate dubbed speech for a job's translation (XTTS v2)")
    sp.add_argument("--job-id", required=True)
    sp.add_argument("--lang", default="hi", help="Language of the translation to speak (default: hi)")
    sp.add_argument("--voice", choices=["clone", "standard"], help="Override tts.voice_mode in config.yaml")
    sp.add_argument("--limit", type=int, help="Only the first N lines (for a quick test)")
    sp.add_argument("--config", type=Path, help="Use a different config file")
    sp.add_argument("--overwrite", action="store_true", help="Generate again, replacing earlier speech")
    sp.add_argument("--redo-flagged", action="store_true",
                    help="Remake only lines that failed the checks last time; keep the rest")
    sp.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    se = sub.add_parser("separate", help="Split the original audio into voices and music/effects (Demucs)")
    se.add_argument("--job-id", required=True)
    se.add_argument("--config", type=Path, help="Use a different config file")
    se.add_argument("--overwrite", action="store_true", help="Separate again")
    se.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    mx = sub.add_parser("mix", help="Mix dubbed speech with the original music/effects and build the dubbed video")
    mx.add_argument("--job-id", required=True)
    mx.add_argument("--lang", default="hi", help="Dubbed language to mix (default: hi)")
    mx.add_argument("--config", type=Path, help="Use a different config file")
    mx.add_argument("--overwrite", action="store_true", help="Mix again, replacing the earlier video")
    mx.add_argument("--verbose", action="store_true", help="Show detailed log messages")

    s = sub.add_parser("subtitles", help="Make a WebVTT (.vtt) subtitle file from a transcript or translation")
    s.add_argument("--job-id", required=True, help="The job to make subtitles for")
    s.add_argument("--lang", help="Language, e.g. 'en' or 'hi' (uses the translation if one exists)")
    s.add_argument("--config", type=Path, help="Use a different config file")
    s.add_argument("--overwrite", action="store_true", help="Replace an existing subtitle file")
    s.add_argument("--verbose", action="store_true", help="Show detailed log messages")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Hindi and other scripts print correctly in the Windows console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config, preset=getattr(args, "preset", None))
        if args.command == "transcribe":
            from dubbing.pipeline.transcribe_job import run_transcription

            if args.language:
                settings = replace(settings, source_language=args.language)
            job_id = validate_job_id(args.job_id or _default_job_id(args.input))
            setup_logging(JobPaths(settings.data_dir, job_id).log_file, verbose=args.verbose)
            out = run_transcription(Path(args.input), job_id, settings, overwrite=args.overwrite)
            print(f"\nDone. Transcript: {out}")
        elif args.command == "script":
            from dubbing.pipeline.script_job import run_script

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            out = run_script(args.job_id, settings, overwrite=args.overwrite)
            print(f"\nDone. Dubbing script: {out}\nReview page: {JobPaths(settings.data_dir, args.job_id).script_review}")
        elif args.command == "translate":
            from dubbing.pipeline.translate_job import run_translation

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            targets = [c.strip() for c in args.to.split(",") if c.strip()] if args.to else None
            outs = run_translation(args.job_id, settings, targets, overwrite=args.overwrite)
            print("\nDone. Translations:\n" + "\n".join(f"  {p}" for p in outs))
        elif args.command == "analyze":
            from dubbing.pipeline.analyze_job import run_analysis

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            out = run_analysis(args.job_id, settings, overwrite=args.overwrite)
            print(f"\nDone. Measurements: {out}")
        elif args.command == "speak":
            from dubbing.pipeline.speech_job import run_speech

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            out = run_speech(args.job_id, settings, lang=args.lang, overwrite=args.overwrite,
                             limit=args.limit, voice_mode=args.voice, redo_flagged=args.redo_flagged)
            print(f"\nDone. Speech manifest: {out}")
        elif args.command == "separate":
            from dubbing.pipeline.separate_job import run_separation

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            out = run_separation(args.job_id, settings, overwrite=args.overwrite)
            print(f"\nDone. Separation report: {out}")
        elif args.command == "mix":
            from dubbing.pipeline.mix_job import run_mix

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            out = run_mix(args.job_id, settings, lang=args.lang, overwrite=args.overwrite)
            print(f"\nDone. Dubbed video: {out}")
        elif args.command == "subtitles":
            from dubbing.pipeline.subtitles_job import run_subtitles

            setup_logging(JobPaths(settings.data_dir, validate_job_id(args.job_id)).log_file, verbose=args.verbose)
            out = run_subtitles(args.job_id, settings, lang=args.lang, overwrite=args.overwrite)
            print(f"\nDone. Subtitles: {out}")
        return 0
    except PipelineError as exc:
        if not logging.getLogger().handlers:
            setup_logging()
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
