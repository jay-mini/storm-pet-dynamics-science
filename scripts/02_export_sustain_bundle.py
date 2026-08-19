from __future__ import annotations

import argparse
from pathlib import Path

from storm_pet.sustain.native_pickle import export_native_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one trusted full-data pySuStaIn pickle to a safe inference bundle."
    )
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--modality", choices=["abeta", "tau"], required=True)
    parser.add_argument("--selected-subtypes", type=int, required=True)
    args = parser.parse_args()
    bundle = export_native_result(
        pickle_path=args.pickle.resolve(),
        training_output=args.training_output.resolve(),
        output_dir=args.output_dir.resolve(),
        model_id=args.model_id,
        modality=args.modality,
        selected_subtypes=args.selected_subtypes,
    )
    print(bundle.root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
