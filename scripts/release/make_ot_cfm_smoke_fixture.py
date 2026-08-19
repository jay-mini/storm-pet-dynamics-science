from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from storm_pet.provenance import write_csv_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic 68-ROI OT-CFM smoke fixture.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rows-per-bin-subtype", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows_per_bin_subtype < 1:
        raise ValueError("rows-per-bin-subtype must be positive")
    rng = np.random.default_rng(args.seed)
    rois = [
        f"CTX_{hemisphere}_{name}_SUVR"
        for hemisphere in ("LH", "RH")
        for name in (
            "BANKSSTS", "CAUDALANTERIORCINGULATE", "CAUDALMIDDLEFRONTAL", "CUNEUS",
            "ENTORHINAL", "FRONTALPOLE", "FUSIFORM", "INFERIORPARIETAL",
            "INFERIORTEMPORAL", "INSULA", "ISTHMUSCINGULATE", "LATERALOCCIPITAL",
            "LATERALORBITOFRONTAL", "LINGUAL", "MEDIALORBITOFRONTAL", "MIDDLETEMPORAL",
            "PARACENTRAL", "PARAHIPPOCAMPAL", "PARSOPERCULARIS", "PARSORBITALIS",
            "PARSTRIANGULARIS", "PERICALCARINE", "POSTCENTRAL", "POSTERIORCINGULATE",
            "PRECENTRAL", "PRECUNEUS", "ROSTRALANTERIORCINGULATE", "ROSTRALMIDDLEFRONTAL",
            "SUPERIORFRONTAL", "SUPERIORPARIETAL", "SUPERIORTEMPORAL", "SUPRAMARGINAL",
            "TEMPORALPOLE", "TRANSVERSETEMPORAL",
        )
    ]
    rows = []
    for subtype in (0, 1):
        for dynamics_bin in range(5):
            for repeat in range(args.rows_per_bin_subtype):
                stage = 0 if dynamics_bin == 0 else dynamics_bin * 4
                row = {
                    "scan_id": f"synthetic-s{subtype}-b{dynamics_bin}-{repeat}",
                    "sustain_subtype": subtype,
                    "sustain_stage": stage,
                    "coarse_stage_bin": np.nan if dynamics_bin == 0 else dynamics_bin - 1,
                    "dynamics_bin": dynamics_bin,
                    "use_in_otfm": True,
                    "subtype_0_prob": 0.9 if subtype == 0 else 0.1,
                    "subtype_1_prob": 0.1 if subtype == 0 else 0.9,
                }
                base = 0.9 + 0.08 * dynamics_bin + 0.02 * subtype
                row.update(
                    {
                        roi: float(base + 0.002 * index + rng.normal(0, 0.005))
                        for index, roi in enumerate(rois)
                    }
                )
                rows.append(row)
    write_csv_atomic(args.output_csv, pd.DataFrame(rows))
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
