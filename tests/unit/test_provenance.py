import json
from pathlib import Path

import pandas as pd

from storm_pet.provenance import sha256_file, write_csv_atomic, write_json_atomic


def test_sha256_file_known_value(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_bytes(b"storm\n")
    assert sha256_file(path) == "855541b77b41974a1b6a8018f40f2dacd606dede9254b32586521087b177645d"


def test_write_json_atomic_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"completed": True, "modality": "tau"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "completed": True,
        "modality": "tau",
    }


def test_write_csv_atomic_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    write_csv_atomic(path, pd.DataFrame({"value": [1, 2]}))
    assert pd.read_csv(path)["value"].tolist() == [1, 2]
