from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from storm_pet.config import load_yaml
from storm_pet.data.features import load_feature_map
from storm_pet.paths import repository_root
from storm_pet.sustain.artifacts import load_sustain_bundle
from storm_pet.sustain.predictor import SustainPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer new scans from a safe SuStaIn bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_sustain_bundle(args.bundle)
    config = load_yaml(args.config)
    feature_map_path = Path(config["feature_map"])
    if not feature_map_path.is_absolute():
        feature_map_path = repository_root() / feature_map_path
    predictor = SustainPredictor(
        bundle=bundle,
        feature_map=load_feature_map(feature_map_path),
        tracer=str(config["tracer"]),
        reference_region=str(config["reference_region"]),
    )
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for index, row in enumerate(rows, start=1):
        scan_id = str(row.pop("scan_id", "") or f"scan-{index}")
        records.append((scan_id, row))
    payload = {
        "model": predictor.schema(),
        "predictions": [item.__dict__ for item in predictor.predict_records(records)],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
