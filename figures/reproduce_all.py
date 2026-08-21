#!/usr/bin/env python
"""Reproduce all main-text figures from the compact source-data packages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
JOBS = {
    "abeta-sustain-main": HERE / "abeta_sustain" / "plot_fig_abeta_sustain_1.py",
    "abeta-sustain-summary": HERE / "abeta_sustain" / "plot_combined_sustain_summary_panel.py",
    "tau-sustain-main": HERE / "tau_sustain" / "plot_fig_tau_sustain_ap_2.py",
    "tau-sustain-summary": HERE / "tau_sustain" / "plot_combined_sustain_summary_panel.py",
    "abeta-dynamics": HERE / "abeta_dynamics" / "plot_fig_abeta_cfm_1.py",
    "tau-dynamics": HERE / "tau_dynamics" / "plot_integrated_dynamics_panel_3x3.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(JOBS),
        help="Run only the named figure jobs (default: run all).",
    )
    args = parser.parse_args()

    selected = args.only or list(JOBS)
    for name in selected:
        script = JOBS[name]
        print(f"\n[{name}] {script.relative_to(HERE)}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)

    print(f"\nReproduced {len(selected)} figure job(s).", flush=True)


if __name__ == "__main__":
    main()
