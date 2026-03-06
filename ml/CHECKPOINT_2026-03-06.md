## Checkpoint

Date: 2026-03-06
Repo: `D:\git\piano_on`

### Run summary (today)

Completed a clean training/eval chain:

1. `retrain-v2`
2. `retrain-v3`
3. `build-ensemble-v23`
4. `eval-holdout` (explicitly using ensemble model)

### Reliability note

`iterate_training.py eval-holdout` defaults to baseline model unless `--model` is passed.
Today, final holdout metrics below are from the explicit command:

- `--model ml/models/piano_detector_v23_ensemble.joblib`
- `--config ml/configs/train_config_v2.yaml`
- `--out-report ml/reports/piano_detector_v23_ensemble_holdout_metrics.json`

### Current best model

- Serving candidate: `v2+v3 ensemble`
- Model path: `ml/models/piano_detector_v23_ensemble.joblib`
- Ensemble build report: `ml/reports/piano_detector_v23_ensemble_build.json`

Ensemble settings:
- method: `weighted_average`
- weights: `v2=0.7`, `v3=0.3`
- decision threshold: `0.84`
- review threshold: `0.72`

### Holdout metrics (80 samples)

From `ml/reports/piano_detector_v23_ensemble_holdout_metrics.json`:

- `threshold = 0.84`
- `auc_roc = 0.979375`
- `auc_pr = 0.980804974345908`
- `fpr = 0.025` (`fp=1/40`)
- `fnr = 0.125` (`fn=5/40`)
- confusion matrix: `tn=39, fp=1, fn=5, tp=35`

### Comparison vs 2026-03-02 checkpoint

Previous reliable checkpoint:
- `auc_roc = 0.8275`
- `auc_pr = 0.8575489484193672`
- `fpr = 0.05`
- `fnr = 0.475`
- `tn=38, fp=2, fn=19, tp=21`

Current run improved all major holdout metrics, especially false negatives:
- `fnr: 0.475 -> 0.125`
- `fpr: 0.05 -> 0.025`

### Fix applied during this run

File changed: `ml/scripts/train_v2.py`

- `ExtraTreesClassifier(n_jobs)` changed from hardcoded `-1` to config-driven:
  - `n_jobs=int(cfg["classifier"].get("n_jobs", 1))`

Reason:
- Training failed with `PermissionError: [WinError 5] Access is denied` in joblib parallel pool setup on this machine.
- Defaulting to single-thread fallback (`1`) restored reliable training execution.

### Operational guidance

- Keep this ensemble as current default candidate.
- If retraining on this environment, keep `v2` at single-thread unless parallel permissions are confirmed stable.
- Continue monitoring missed-piano errors on real home recordings; current holdout is much healthier but not perfect.
