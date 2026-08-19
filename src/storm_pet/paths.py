from __future__ import annotations

import os
from pathlib import Path

from storm_pet.exceptions import ConfigurationError


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ConfigurationError(
            f"Required environment variable {name} is not set. "
            f"PowerShell example: $env:{name}='D:\\path\\to\\authorized-data'"
        )
    return Path(value).expanduser().resolve()


def ensure_within(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ConfigurationError(f"Path escapes configured root: {resolved_path}")
    return resolved_path

