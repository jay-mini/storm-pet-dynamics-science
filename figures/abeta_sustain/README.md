# Publication figure source data

This directory contains only the compact numerical data and standard surface
geometry required to redraw the two requested figures. It does not contain the
large analysis tables, intermediate statistics, or copies of the old PNG/JPG
panels.

## Reproduce the figures

Run from this directory with the project Python environment:

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe plot_fig_abeta_sustain_1.py
D:\Anaconda\envs\brain_dynamic\python.exe plot_combined_sustain_summary_panel.py
```

Outputs are written to `outputs/`.

Required Python packages: `numpy`, `pandas`, `matplotlib`, `nilearn`, and
`statsmodels`.

## Data inventory

`data/fig_abeta_sustain_1/`

- `dk68_stage_adjusted_spatial_residuals.csv`: 68 plotted DK residual values.
- `dk68_stage_mmse_apoe4_points.csv`: de-identified point-level stage, MMSE,
  and APOE4 carrier data used by the upper panels. Repeated `subject_id` values
  preserve clustering for the MMSE standard errors.
- `tau_pet_lh_stage_mean_values.csv`: the 24 LH DK-34 stage-mean maps displayed
  in the two 2-by-6 Tau-PET blocks.
- `tau_pet_display_order.csv`: the exact subtype, stage, and panel order.
- `tau_pet_color_scale.csv`: the shared Tau-PET color limits and colormap.
- `fsaverage5_surface_data.npz`: standard fsaverage5 mesh, sulcal background,
  and Desikan label lookup required to map the CSV values to the cortex.

The reference layout uses 12 displayed maps per subtype. The selected stages
are recorded explicitly in `tau_pet_display_order.csv`. Subtype 2 has no stage
18 mean map in the supplied stage-map data, so stage 17 is used in that display
position; no value was interpolated or fabricated.

`data/combined_sustain_summary_panel/`

- `subtype_probability_heatmap.csv`: the sorted posterior probabilities shown
  in the subtype heatmap.
- `longitudinal_first_followup_pairs.csv`: the first longitudinal pair per
  participant, with only the variables needed for the transition heatmap,
  stage scatter, and stability probability panel.
- `scan_stage_groups.csv`: stage and the two grouping variables needed for the
  Aβ and research-group raincloud plots.

All row identifiers in this package are plotting row numbers or newly assigned
de-identified subject labels; the original large input tables are not required
by either plotting script.
