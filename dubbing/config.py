"""Loads config.yaml into a typed Settings object and resolves presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dubbing.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

VALID_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(frozen=True)
class AsrSettings:
    model: str
    beam_size: int
    vad_filter: bool
    word_timestamps: bool


@dataclass(frozen=True)
class TranslationSettings:
    model: str
    batch_size: int
    num_beams: int
    max_new_tokens: int


@dataclass(frozen=True)
class Settings:
    preset: str
    source_language: str | None
    target_languages: list[str]
    device: str
    data_dir: Path
    ffmpeg_path: str | None
    asr: AsrSettings
    translation: TranslationSettings


def load_settings(config_path: Path | None = None, preset: str | None = None) -> Settings:
    """Read config.yaml. `preset` overrides the preset named in the file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc
    return _build_settings(raw, preset, base_dir=path.parent)


def _build_settings(raw: dict[str, Any], preset_override: str | None, base_dir: Path) -> Settings:
    presets = raw.get("presets") or {}
    preset = preset_override or raw.get("preset", "fast")
    if preset not in presets:
        raise ConfigError(f"Unknown preset '{preset}'. Choose one of: {', '.join(presets) or '(none defined)'}")

    device = raw.get("device", "auto")
    if device not in VALID_DEVICES:
        raise ConfigError(f"device must be one of {sorted(VALID_DEVICES)}, got '{device}'")

    paths = raw.get("paths") or {}
    data_dir = Path(paths.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    asr_raw = raw.get("asr") or {}
    model = presets[preset].get("asr_model")
    if not model:
        raise ConfigError(f"Preset '{preset}' does not set asr_model")

    tr_raw = raw.get("translation") or {}
    translation = TranslationSettings(
        model=presets[preset].get("translation_model") or tr_raw.get("model", "facebook/nllb-200-distilled-600M"),
        batch_size=int(tr_raw.get("batch_size", 8)),
        num_beams=int(tr_raw.get("num_beams", 4)),
        max_new_tokens=int(tr_raw.get("max_new_tokens", 256)),
    )
    if translation.batch_size < 1 or translation.num_beams < 1:
        raise ConfigError("translation.batch_size and translation.num_beams must be 1 or more")

    return Settings(
        preset=preset,
        source_language=raw.get("source_language"),
        target_languages=list(raw.get("target_languages") or []),
        device=device,
        data_dir=data_dir,
        ffmpeg_path=paths.get("ffmpeg"),
        asr=AsrSettings(
            model=model,
            beam_size=int(asr_raw.get("beam_size", 5)),
            vad_filter=bool(asr_raw.get("vad_filter", True)),
            word_timestamps=bool(asr_raw.get("word_timestamps", True)),
        ),
        translation=translation,
    )
