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


VOICE_MODES = {"clone", "standard"}


@dataclass(frozen=True)
class TtsSettings:
    accept_coqui_license: bool
    model: str
    voice_mode: str
    standard_speaker: str
    min_reference_s: float
    max_borrow_distance_s: float
    max_speed: float
    temperature: float
    seed: int
    verify_with_asr: bool
    clarity_fallback: bool


@dataclass(frozen=True)
class Settings:
    preset: str
    source_language: str | None
    target_languages: list[str]
    device: str
    data_dir: Path
    ffmpeg_path: str | None
    ffmpeg_dll_dir: str | None
    asr: AsrSettings
    translation: TranslationSettings
    tts: TtsSettings


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

    t = raw.get("tts") or {}
    tts = TtsSettings(
        accept_coqui_license=bool(t.get("accept_coqui_license", False)),
        model=t.get("model", "tts_models/multilingual/multi-dataset/xtts_v2"),
        voice_mode=t.get("voice_mode", "clone"),
        standard_speaker=t.get("standard_speaker", "Damien Black"),
        min_reference_s=float(t.get("min_reference_s", 3.0)),
        max_borrow_distance_s=float(t.get("max_borrow_distance_s", 10.0)),
        max_speed=float(t.get("max_speed", 1.3)),
        temperature=float(t.get("temperature", 0.7)),
        seed=int(t.get("seed", 1234)),
        verify_with_asr=bool(t.get("verify_with_asr", True)),
        clarity_fallback=bool(t.get("clarity_fallback", True)),
    )
    if tts.voice_mode not in VOICE_MODES:
        raise ConfigError(f"tts.voice_mode must be one of {sorted(VOICE_MODES)}, got '{tts.voice_mode}'")
    if not 1.0 <= tts.max_speed <= 2.0:
        raise ConfigError("tts.max_speed must be between 1.0 and 2.0")

    return Settings(
        preset=preset,
        source_language=raw.get("source_language"),
        target_languages=list(raw.get("target_languages") or []),
        device=device,
        data_dir=data_dir,
        ffmpeg_path=paths.get("ffmpeg"),
        ffmpeg_dll_dir=paths.get("ffmpeg_dll_dir"),
        asr=AsrSettings(
            model=model,
            beam_size=int(asr_raw.get("beam_size", 5)),
            vad_filter=bool(asr_raw.get("vad_filter", True)),
            word_timestamps=bool(asr_raw.get("word_timestamps", True)),
        ),
        translation=translation,
        tts=tts,
    )
