# Integrated dynamics visualization: figure source data

This folder reproduces `integrated_dynamics_visualization_panel_3x3.png` without loading model outputs, clinical records, subject identifiers, or analysis-intermediate files.

## Contents

- `plot_integrated_dynamics_panel_3x3.py`: standalone plotting script.
- `data/reconstruction_mse.csv`: reconstruction-MSE values for the histogram.
- `data/roi_absolute_error_heatmap.csv`: displayed 2-subtype × 5-stage-group × 68-ROI heatmap cells.
- `data/umap_background_points.csv`: UMAP coordinates, stage bins, and displayed stage labels shared by the three UMAP panels.
- `data/velocity_vectors.csv`: displayed velocity origins and arrow vectors.
- `data/sample_trajectories.csv`: displayed sample-trajectory coordinates and ordering fields.
- `data/mean_trajectories.csv`: displayed subtype mean-trajectory coordinates and ordering fields.
- `data/trajectory_anchors.csv`: shared-root and subtype branch-anchor coordinates.
- `data/decoded_mean_trajectories.csv`: decoded mean-path coordinates and ordering fields.
- `data/individual_global_change_predictions.csv`: displayed individual true/predicted global tau changes.
- `data/population_forecast_metrics.csv`: displayed population forecast coordinates and the three distribution-distance metrics for the six compared methods.
- `outputs/integrated_dynamics_visualization_panel_3x3.png`: reproduced figure.

The CSV files contain only values used directly to draw the figure. Subject IDs, dates, diagnoses, raw ROI observations, latent representations, training outputs, and non-displayed validation columns are intentionally excluded.

## Reproduce

From this directory, run:

```powershell
python plot_integrated_dynamics_panel_3x3.py
```

Required Python packages: `numpy`, `pandas`, and `matplotlib`.
