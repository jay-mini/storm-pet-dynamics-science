from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from storm_pet.paths import repository_root
from storm_pet.provenance import write_csv_atomic
from storm_pet.sustain.native_pickle import load_numpy_result
from storm_pet.sustain.training import prepare_sustain_training_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach selected full-data SuStaIn assignments to the retained training scans."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    _, prepared = prepare_sustain_training_data(
        args.input_csv.resolve(), args.config.resolve(), repository_root()
    )
    frame = pd.read_csv(args.input_csv).iloc[list(prepared.retained_row_numbers)].reset_index(drop=True)
    result = load_numpy_result(args.pickle.resolve())
    columns = {
        "sustain_subtype": "ml_subtype",
        "sustain_subtype_prob": "prob_ml_subtype",
        "sustain_stage": "ml_stage",
        "sustain_stage_prob": "prob_ml_stage",
    }
    for destination, source in columns.items():
        if source not in result:
            raise KeyError(f"SuStaIn pickle is missing {source}")
        values = np.asarray(result[source]).reshape(-1)
        if len(values) != len(frame):
            raise ValueError("SuStaIn result row count differs from retained training scans")
        frame[destination] = values
    probability = np.asarray(result.get("prob_subtype"))
    if probability.ndim == 2 and probability.shape[0] == len(frame):
        for subtype in range(probability.shape[1]):
            frame[f"subtype_{subtype}_prob"] = probability[:, subtype]
    write_csv_atomic(args.output_csv, frame)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
