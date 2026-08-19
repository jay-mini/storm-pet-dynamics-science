from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storm_pet.data.pipeline import prepare_abeta_gmm_stage, prepare_tau_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Aβ or Tau data for the STORM pipeline.")
    subparsers = parser.add_subparsers(dest="modality", required=True)
    abeta = subparsers.add_parser("abeta")
    abeta.add_argument("--input-csv", type=Path, required=True)
    abeta.add_argument("--output-dir", type=Path, required=True)

    tau = subparsers.add_parser("tau")
    tau.add_argument("--tau-csv", type=Path, required=True)
    tau.add_argument("--diagnosis-csv", type=Path, required=True)
    tau.add_argument("--abeta-status-csv", type=Path, required=True)
    tau.add_argument("--demographic-csv", type=Path, required=True)
    tau.add_argument("--output-dir", type=Path, required=True)
    tau.add_argument("--tracer", default="FTP")
    tau.add_argument("--max-difference-days", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.modality == "abeta":
        result = prepare_abeta_gmm_stage(args.input_csv, args.output_dir)
    else:
        result = prepare_tau_stage(
            args.tau_csv,
            args.diagnosis_csv,
            args.abeta_status_csv,
            args.demographic_csv,
            args.output_dir,
            tracer=args.tracer,
            max_difference_days=args.max_difference_days,
        )
    print(result.final_table)
    print(result.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

