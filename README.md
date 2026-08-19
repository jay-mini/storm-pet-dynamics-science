# STORM PET dynamics — scientific release

This release contains the scientific Aβ/Tau pipeline only: authorized-data preparation,
full-data z-score SuStaIn training and inference artifacts, common stage binning, and OT-CFM
training/trajectory reconstruction. The web application, FastAPI service, deployment registry,
participant data, pretrained weights, local audit records, CV models, and Appendix ablations are
deliberately absent.

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[sustain,ot-cfm,dev]"
```

pySuStaIn is pinned to official commit
`708fa22d89c9692dbeafcc2c31a1c8460ced5640`. Native pickle files are training artifacts; never
load an untrusted pickle. The inference bundle produced by this project is JSON/NumPy based and
uses `allow_pickle=False`.

## Main-text workflow

1. Obtain access to the source cohort data and keep it under `data/authorized/` (ignored by Git).
2. Prepare Aβ/Tau scan tables with `scripts/01_prepare_data.py`.
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

## Scope and release status

This tree is created from an explicit allowlist and has passed a local secret/path/artifact scan.
It is not ready to publish until the authors choose a source-code license and replace
`LICENSE_DECISION_REQUIRED.md` with that license. See `docs/OPEN_SOURCE_CHECKLIST_CN.md`.
