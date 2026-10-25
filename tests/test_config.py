from pathlib import Path

import pytest

from dubbing.config import DEFAULT_CONFIG_PATH, load_settings
from dubbing.errors import ConfigError, PipelineError
from dubbing.paths import validate_job_id


def test_default_config_loads():
    settings = load_settings()
    assert settings.preset in {"fast", "quality"}
    assert settings.asr.model
    assert settings.data_dir.is_absolute()


def test_preset_override_changes_model():
    assert load_settings(preset="fast").asr.model == "medium"
    assert load_settings(preset="quality").asr.model == "large-v3"


def test_unknown_preset_is_rejected():
    with pytest.raises(ConfigError, match="Unknown preset"):
        load_settings(preset="turbo")


def test_invalid_device_is_rejected(tmp_path: Path):
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("device: auto", "device: gpu")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="device"):
        load_settings(cfg)


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize("bad", ["", "has space", "../escape", "a" * 65])
def test_bad_job_ids_are_rejected(bad):
    with pytest.raises(PipelineError):
        validate_job_id(bad)


def test_good_job_id():
    assert validate_job_id("job_001") == "job_001"
