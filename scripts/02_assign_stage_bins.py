from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from storm_pet.data.stage_bins import assign_stage_bins
from storm_pet.provenance import write_csv_atomic


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign common positive-stage bins and the stage-0 OT-CFM root."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--definition-csv", type=Path, required=True)
    parser.add_argument("--positive-bins", type=int, default=4)
    parser.add_argument("--minimum-samples", type=int, default=25)
    parser.add_argument(
        "--stage-ranges",
        help="Comma-separated fixed main-text ranges, for example 1-2,3-6,7-15,16-20.",
    )
    args = parser.parse_args()
    fixed_ranges = None
    if args.stage_ranges:
        fixed_ranges = tuple(
            tuple(int(value) for value in item.split("-", maxsplit=1))
            for item in args.stage_ranges.split(",")
        )
    output, definitions = assign_stage_bins(
        pd.read_csv(args.input_csv),
        desired_positive_bins=args.positive_bins,
        minimum_samples_per_bin=args.minimum_samples,
        fixed_stage_ranges=fixed_ranges,
    )
    write_csv_atomic(args.output_csv, output)
    write_csv_atomic(args.definition_csv, definitions)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
