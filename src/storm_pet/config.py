from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from storm_pet.exceptions import ConfigurationError

ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise ConfigurationError(f"Configuration requires environment variable {name}")
        return resolved

    return ENVIRONMENT_PATTERN.sub(replace, value)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return _resolve_environment(payload)


def require_keys(payload: dict[str, Any], required: set[str], context: str) -> None:
    missing = required.difference(payload)
    if missing:
        raise ConfigurationError(f"{context} is missing required keys: {sorted(missing)}")


def reject_unknown_keys(payload: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ConfigurationError(f"{context} contains unknown keys: {sorted(unknown)}")


def config_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

