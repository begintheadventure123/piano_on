## Checkpoint

Date: 2026-03-02
Repo: `D:\git\piano_on`

### Current reliable state

- Default serving/test model is `v2+v3 ensemble`.
- Current ensemble model file:
  - `ml/models/piano_detector_v23_ensemble.joblib`
- Latest reliable holdout metrics for the current ensemble:
  - `threshold = 0.84`
  - `auc_roc = 0.8275`
  - `auc_pr = 0.8575489484193672`
  - `fpr = 0.05`
  - `fnr = 0.475`
  - confusion matrix: `tn=38, fp=2, fn=19, tp=21`
- Metrics source:
  - `ml/reports/piano_detector_v23_ensemble_holdout_metrics.json`

### Important findings

- Baseline is retired as the main path.
- `v2` and `v3` were built, and `v2+v3 ensemble` is now the default inference path.
- Main remaining weakness is still false negatives on real piano.
- The 19 missed piano holdout clips are concentrated in `home_piano_review_top60_v2_labeled`.
- Feature-gap analysis indicates missed piano clips tend to be:
  - lower RMS / weaker
  - brighter / thinner / noisier
  - lower harmonic ratio

Relevant reports:

- `ml/reports/piano_detector_v23_ensemble_false_negatives.csv`
- `ml/reports/piano_detector_v23_ensemble_false_negatives_summary.json`
- `ml/reports/piano_detector_v23_ensemble_piano_gap_features.csv`
- `ml/reports/piano_detector_v23_ensemble_piano_gap_summary.json`

### Data / labeling state

- Managed asset pool is active under:
  - `ml/data/review/asset_pool`
- Review labels are DB-backed.
- `Promote Current Labels` writes reviewed clips into managed training assets.
- User already reviewed and promoted:
  - the hard-positive queue
  - the 19 missed-piano holdout queue

### UI / workflow state

- Upload page supports preset label for raw files:
  - no preset
  - all non-piano
  - all piano
- Batch/test/review now default to the ensemble model.
- `Retrain Model` on the `v2+v3 Ensemble` track was fixed to run:
  1. retrain `v2`
  2. retrain `v3`
  3. rebuild ensemble
- Timestamps shown in UI were changed to local time instead of raw UTC display.

### Training config change made tonight

- `ml/configs/train_config_v2.yaml`
  - `classifier.n_estimators` reduced from `700` to `350`

Reason:

- `v2` retraining was too slow for iterative use.
- We were attempting to rerun a clean `v2 -> v3 -> ensemble -> holdout` chain with the lighter `v2` config.

### Interrupted work

- A fresh `retrain-v2` attempt was started after cleaning stale processes.
- That run was interrupted before completion.
- After the interruption, all residual Python training processes were stopped.
- As of this checkpoint, there should be no intentionally running training job.

Model timestamps at checkpoint time:

- `ml/models/piano_detector_v2.joblib` last reliable write: `2026-03-02 9:31 PM`
- `ml/models/piano_detector_v3.joblib` last reliable write: `2026-03-02 9:41 PM`
- `ml/models/piano_detector_v23_ensemble.joblib` last reliable write: `2026-03-02 9:41 PM`

### Resume plan for tomorrow

1. Verify no stale Python training processes are running.
2. Run a clean sequential retrain:
   - `.\.venv-ml\Scripts\python.exe ml\scripts\iterate_training.py retrain-v2`
   - `.\.venv-ml\Scripts\python.exe ml\scripts\iterate_training.py retrain-v3`
   - `.\.venv-ml\Scripts\python.exe ml\scripts\iterate_training.py build-ensemble-v23`
3. Re-run holdout on the rebuilt ensemble.
4. Compare against the current reliable ensemble metrics:
   - `auc_roc 0.8275`
   - `fpr 0.05`
   - `fnr 0.475`
5. Decide whether the lighter `v2` config helped, or whether the next move should be:
   - threshold/weight tuning
   - a stronger model backbone

### Notes

- The goal remains: a final accurate piano-sound model, not just fixing one issue.
- Do not trust partial model writes from interrupted retrains.
- Prefer one clean training chain at a time; avoid concurrent retrains and duplicate servers.
