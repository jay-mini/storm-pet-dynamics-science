from __future__ import annotations

import argparse
from pathlib import Path

from storm_pet import __version__


def _train_sustain(args: argparse.Namespace) -> int:
    from storm_pet.paths import repository_root
    from storm_pet.sustain.training import run_sustain_training

    path = run_sustain_training(
        input_csv=args.input_csv,
        config_path=args.config,
        output_dir=args.output_dir,
        repository_root=repository_root(),
        dataset_name=args.dataset_name,
        seed=args.seed,
        use_parallel_startpoints=not args.no_parallel,
    )
    print(path)
    return 0


def _train_ot_cfm(args: argparse.Namespace) -> int:
    from storm_pet.ot_cfm.config import load_ot_cfm_config
    from storm_pet.ot_cfm.pipeline import main as run
    from storm_pet.paths import repository_root

    _, argv = load_ot_cfm_config(args.config, repository_root())
    if args.smoke_test:
        output_index = argv.index("--out_dir") + 1
        output = Path(argv[output_index])
        argv[output_index] = str(output.with_name(output.name + "_smoke"))
        argv.extend(["--ae_epochs", "1", "--dry_run_ae_only"])
    run(argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="storm-pet-science")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    sustain = commands.add_parser("train-sustain")
    sustain.add_argument("--config", type=Path, required=True)
    sustain.add_argument("--input-csv", type=Path, required=True)
    sustain.add_argument("--output-dir", type=Path, required=True)
    sustain.add_argument("--dataset-name")
    sustain.add_argument("--seed", type=int)
    sustain.add_argument("--no-parallel", action="store_true")
    sustain.set_defaults(handler=_train_sustain)
    ot_cfm = commands.add_parser("train-ot-cfm")
    ot_cfm.add_argument("--config", type=Path, required=True)
    ot_cfm.add_argument("--smoke-test", action="store_true")
    ot_cfm.set_defaults(handler=_train_ot_cfm)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.handler(args))
