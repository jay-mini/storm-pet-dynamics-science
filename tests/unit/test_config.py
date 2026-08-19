from pathlib import Path

import pytest

from storm_pet.config import config_sha256, load_yaml
from storm_pet.exceptions import ConfigurationError


def test_load_yaml_resolves_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORM_TEST_ROOT", "D:/authorized")
    path = tmp_path / "config.yaml"
    path.write_text('data_root: "${STORM_TEST_ROOT}/tau"\n', encoding="utf-8")
    assert load_yaml(path)["data_root"] == "D:/authorized/tau"


def test_load_yaml_fails_when_environment_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('data_root: "${STORM_MISSING_ROOT}"\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="STORM_MISSING_ROOT"):
        load_yaml(path)


def test_config_hash_is_key_order_independent() -> None:
    assert config_sha256({"a": 1, "b": 2}) == config_sha256({"b": 2, "a": 1})

