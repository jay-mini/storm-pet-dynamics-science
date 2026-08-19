from __future__ import annotations

import argparse
from pathlib import Path

from storm_pet.ot_cfm.pipeline import main as run_ot_cfm
from storm_pet.ot_cfm.config import load_ot_cfm_config
from storm_pet.paths import repository_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the unified Aβ/Tau OT-CFM pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override training to one autoencoder epoch and stop before CFM training.",
    )
    args = parser.parse_args()
    _, argv = load_ot_cfm_config(args.config, repository_root())
    if args.smoke_test:
        output_index = argv.index("--out_dir") + 1
        smoke_dir = Path(argv[output_index]).with_name(Path(argv[output_index]).name + "_smoke")
        argv[output_index] = str(smoke_dir)
        argv.extend(["--ae_epochs", "1", "--dry_run_ae_only"])
    run_ot_cfm(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
