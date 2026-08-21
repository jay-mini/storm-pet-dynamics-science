# STORM PET dynamics — scientific release

This release contains the scientific Aβ/Tau pipeline: authorized-data preparation, full-data z-score SuStaIn training and inference artifacts, common stage binning, and OT-CFM training and trajectory reconstruction. In addition, the companion STORM-PET web platform (www.storm-pet.xyz) provides an end-to-end interface through which users can upload compatible Aβ or tau PET data and apply the pretrained models for subtype/stage inference, trajectory prediction, and downstream dynamical analysis.

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[sustain,ot-cfm,dev]"
```

pySuStaIn is pinned to official commit
`708fa22d89c9692dbeafcc2c31a1c8460ced5640`. Native pickle files are training artifacts; never
load an untrusted pickle. The inference bundle produced by this project is JSON/NumPy based and
uses `allow_pickle=False`. This repository uses pySuStaIn for subtype and stage inference. Please also cite the original SuStaIn methodology and the pySuStaIn software when using this component. Details see https://doi.org/10.1038/s41467-018-05892-0 .

## Main-text workflow

1. Obtain access to the source cohort data and keep it under `data/authorized/` (ignored by Git).
2. Prepare Aβ/Tau scan tables with `scripts/01_prepare_data.py`. Aβ status is defined once as
   `CENTILOIDS > 18`; values equal to 18 are negative. The output preserves the direct
   `ABETA_CL_LABEL` and the participant-level harmonized `ABETA_CL_LABEL_monotonic` used for
   SuStaIn control selection and Tau matching.
3. Train the full-data SuStaIn model. The main-text configs stop at the selected two-subtype
model, so they do not spend time training unused 3–5 subtype alternatives:

```powershell
python scripts/02_train_sustain.py --config configs/sustain/tau_paper.yaml `
  --input-csv data/authorized/tau_scan_table.csv --output-dir outputs/tau_sustain
```

4. Use the selected full-data native result (the current paper selects two subtypes) to export a
safe inference bundle and extract scan assignments:

```powershell
python scripts/02_export_sustain_bundle.py --pickle <selected-full-data.pickle> `
  --training-output outputs/tau_sustain --output-dir outputs/tau_sustain_bundle `
  --model-id tau-paper-v1 --modality tau --selected-subtypes 2
python scripts/02_extract_sustain_results.py --input-csv data/authorized/tau_scan_table.csv `
  --config configs/sustain/tau_paper.yaml --pickle <selected-full-data.pickle> `
  --output-csv outputs/tau_sustain_assignments.csv
```

5. Assign the four positive-stage bins plus the shared stage-0 root:

```powershell
python scripts/02_assign_stage_bins.py --input-csv outputs/tau_sustain_assignments.csv `
  --output-csv data/authorized/tau/ot_cfm_input.csv `
  --definition-csv outputs/tau_stage_bins.csv --stage-ranges "1-2,3-6,7-15,16-20"
```

6. Train OT-CFM and reproduce its main trajectory outputs:

```powershell
python scripts/03_train_ot_cfm.py --config configs/ot_cfm/tau_paper.yaml
```

For Aβ use `--stage-ranges "1-3,4-8,9-20,21-23,24-24"`. These ranges are frozen from the
main-text checkpoint rather than recomputed from a new cohort. Run the same sequence with the Aβ
configurations. Full SuStaIn training is expensive; unit tests
validate contracts without running MCMC:

```powershell
python -m pytest tests/unit
python scripts/release/make_ot_cfm_smoke_fixture.py --output-csv data/authorized/tau/ot_cfm_input.csv
python scripts/03_train_ot_cfm.py --config configs/ot_cfm/tau_paper.yaml --smoke-test
```

The synthetic CSVs in `data/demo/` test new-scan input shape; they are not a substitute for an
authorized training cohort and cannot reproduce paper estimates.

## Main-text figures

Compact figure source data, plotting code, and reference exports are in [`figures/`](figures/).
Install the figure dependencies and reproduce every included main-text figure with:

```powershell
python -m pip install -e ".[figures]"
python figures/reproduce_all.py
```

## Scope and release status

This tree is created from an explicit allowlist and has passed a local secret/path/artifact scan.
The source code is released under the [MIT License](LICENSE). Please cite this software using
[`CITATION.cff`](CITATION.cff).

The MIT License applies only to the source code and documentation authored for this repository.
It does not grant access to, redistribute, or license participant-level data, native SuStaIn
pickle files, pretrained model weights, private deployment bundles, or web-service components.
Those materials are not included here and remain governed by their applicable data-use agreements
and release policies. See `DATA_AVAILABILITY.md`, `MODEL_RELEASE_POLICY.md`, and
`docs/OPEN_SOURCE_CHECKLIST_CN.md`.
