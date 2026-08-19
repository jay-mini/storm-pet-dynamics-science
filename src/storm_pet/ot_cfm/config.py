from __future__ import annotations

from pathlib import Path
from typing import Any

from storm_pet.config import load_yaml, require_keys


def load_ot_cfm_config(path: str | Path, repository_root: str | Path) -> tuple[dict[str, Any], list[str]]:
    config_path = Path(path)
    payload = load_yaml(config_path)
    require_keys(payload, {"schema_version", "modality", "input_csv", "output_dir", "parameters"}, "OT-CFM config")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported OT-CFM config schema version")
    root = Path(repository_root).resolve()
    input_csv = Path(payload["input_csv"])
    output_dir = Path(payload["output_dir"])
    if not input_csv.is_absolute():
        input_csv = root / input_csv
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    argv = ["--input_csv", str(input_csv.resolve()), "--out_dir", str(output_dir.resolve())]
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("OT-CFM parameters must be a mapping")
    for key, value in parameters.items():
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(option)
            else:
                negative = f"--no_{key}"
                argv.append(negative)
        elif value is not None:
            argv.extend([option, str(value)])
    return payload, argv

