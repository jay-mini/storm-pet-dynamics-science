# CFM publication figure source data

This folder contains a self-contained plotting-data package and one plotting
script for the following outputs:

- `integrated_dynamics_visualization_panel_2x3.png`
- `overview_2ptid_observed_then_decoded_lh_lateral_2x8.png`
- `fig_abeta_cfm_1.jpg`

The normal plotting path reads only files in `data/`. It does not load trained
models, checkpoints, the full clinical cohort, or pre-rendered source images.
The 021 subject is retained only because the combined figure's bottom panel
contains both CN (021) and MCI (041) rows.

Run from the repository root:

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe OT_CFM_Visualization\CFM_Publication_Figure_Source_Data\plot_fig_abeta_cfm_1.py
```

`--prepare-data` is a provenance helper used to export the compact package once
from this repository's analysis outputs. It is not needed after `data/` has
been prepared.
