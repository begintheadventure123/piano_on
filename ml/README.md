# ML Training Pipeline

This folder contains the offline training/evaluation pipeline for piano activity detection.

Canonical workflow and source-of-truth notes live in `ml/ARCHITECTURE.md`.

## Folder layout
- `ml/data/raw/piano`: positive clips
- `ml/data/raw/non_piano`: negative clips
- `ml/data/raw/mixed`: mixed clips (currently treated as positive for clip-level baseline)
- `ml/data/labels/manifest.csv`: generated file manifest
- `ml/models`: trained models
- `ml/reports`: metrics outputs
- `ml/scripts`: setup/download/train/predict scripts

## Quick start
1. Setup environment:
   - `powershell -ExecutionPolicy Bypass -File ml/scripts/setup_env.ps1`
2. Activate:
   - `.\.venv-ml\Scripts\Activate.ps1`
3. Download starter piano samples (UIowa):
   - `python ml/scripts/download_uiowa_piano.py --limit 60`
4. Download starter non-piano samples (UIowa instruments/percussion/objects):
   - `python ml/scripts/download_uiowa_non_piano.py --limit 180 --include-2012`
5. (Optional) Add your own non-piano clips under `ml/data/raw/non_piano`.
6. Build manifest:
   - `python ml/scripts/prepare_manifest.py`
7. Train baseline:
   - `python ml/scripts/train_baseline.py`
8. Train stronger v2 candidate:
   - `python ml/scripts/train_v2.py`
9. Predict one file:
   - `python ml/scripts/predict_file.py path\\to\\audio.m4a`
10. Estimate piano timeline and total piano duration in a long file:
   - `python ml/scripts/estimate_piano_timeline.py path\\to\\long_recording.m4a`
   - Outputs:
     - frame-level probabilities CSV
     - merged piano intervals CSV
     - summary JSON with total estimated piano seconds

## Utility: split one long recording into clips
- Example (non-piano, 8s clips, no overlap):
  - `python ml/scripts/split_long_audio.py --input path\\to\\long_recording.m4a --out ml/data/raw/non_piano/recorded --label-prefix non_piano --clip-seconds 8 --hop-seconds 8`

## Long-audio timeline estimation
Use this when the goal is "how long was piano played in this recording?" rather than clip-level classification.

- Recommended starting point:
  - `python ml/scripts/estimate_piano_timeline.py path\\to\\long_recording.m4a`
- Default behavior:
  - scans `2.0s` windows every `0.5s`
  - smooths probabilities across neighboring frames
  - uses hysteresis (`enter-threshold` / `exit-threshold`) to produce stable piano-active intervals
  - merges short gaps and reports total estimated piano duration
- Useful knobs:
  - `--window-seconds 2`
  - `--hop-seconds 0.5`
  - `--enter-threshold 0.70`
  - `--exit-threshold 0.60`
  - `--merge-gap-seconds 1.0`

## Long-audio timeline evaluation
Use this when you have manual piano time intervals for a long recording and want to measure duration error and overlap quality.

- Annotation template:
  - `ml/data/eval/long_audio_annotations_template.csv`
- Suggested first-pass annotation queue:
  - `ml/data/eval/long_audio_annotation_queue.csv`
- Required CSV columns:
  - `audio_path,start_seconds,end_seconds,label`
- Example:
  - `python ml/scripts/evaluate_piano_timeline.py path\\to\\long_recording.m4a --annotations-csv ml\\data\\eval\\my_long_audio_annotations.csv`
- Outputs:
  - predicted frame probabilities CSV
  - predicted piano intervals CSV
  - normalized ground-truth intervals CSV
  - summary JSON with:
    - predicted vs ground-truth piano seconds
    - absolute duration error
    - precision / recall / F1 / IoU over timeline coverage

## Iterative review loop
Use this when model testing finds false positives and you want to keep improving precision on non-piano audio.

1. Run one inbox batch (split + score + persist review queue to DB):
   - `python ml/scripts/iterate_training.py run-batch --batch ml/data/inbox/unseen_batch_001`
2. Review flagged clips in local web UI:
   - `python ml/tools/clip_reviewer/server.py --root .`
   - Labels auto-save to `ml/data/ops/activity.db`
3. Promote reviewed labels into training data:
   - `python ml/scripts/iterate_training.py promote-labels --batch ml/data/inbox/unseen_batch_001 --bucket unseen_batch_001`
4. Retrain:
   - `python ml/scripts/iterate_training.py retrain`
5. Retrain v2 candidate:
   - `python ml/scripts/iterate_training.py retrain-v2`

Fast path for known all-non-piano batches:
- `python ml/scripts/iterate_training.py run-batch --batch ml/data/inbox/unseen_batch_001 --auto-promote-all-non-piano --bucket unseen_batch_001`
- Then retrain with `python ml/scripts/iterate_training.py retrain`

New reliability utilities:
- Holdout evaluation (fixed dataset, never used for training):
  - `python ml/scripts/iterate_training.py eval-holdout --holdout-root ml/data/eval/holdout`
- Build fixed holdout from manually labeled hard examples:
  - `python ml/scripts/archive_manual_labels.py`
  - `python ml/scripts/build_manual_holdout.py`
- Dataset audit (suspect labels/conflicts):
  - `python ml/scripts/iterate_training.py audit-dataset`
- Multi-model probability fusion from score CSV files:
  - `python ml/scripts/iterate_training.py combine-scores --scores ml/reports/clip_scores_a.csv,ml/reports/clip_scores_b.csv --weights 0.5,0.5`

## One-place dashboard (recommended)
Use this to avoid manually running multiple scripts.

1. Start dashboard:
   - `python ml/tools/clip_reviewer/server.py --root .`
2. Open:
   - `http://127.0.0.1:8765`
3. In UI:
   - Upload many long files with `Upload Files To Batch` (optional, recommended)
   - `Run Batch + Load Review`
   - Label clips (auto-saved to DB)
   - `Finalize Batch (Auto)` (recommended end-to-end: promote + retrain + cleanup)

## Activity DB (new)
- SQLite is now the system of record at `ml/data/ops/activity.db`.
- Uploads are tracked with SHA-256 and deduplicated by content hash (not just filename).
- Review queues and review labels are persisted to DB by batch.
- Promotion dedup (`promote-labels` and auto-promote mode) now uses this same DB.
- One-time migration for historical inbox review CSVs:
  - `python ml/scripts/migrate_review_csvs_to_db.py`
- Dashboard `Train & Evaluate -> Storage Intelligence` now includes:
  - upload history for big files + duplicates,
  - ML storage analysis (folder sizes, top large files, inbox batch footprint),
  - one-click smart cleanup (safe cache/temp cleanup + retention policy).

## Manual label safety
- Manual labels are now archived to `ml/data/review/manual_label_archive`.
- Run `python ml/scripts/archive_manual_labels.py` after important review sessions to snapshot:
  - consolidated human labels,
  - inbox/report label files,
  - DB review label/event tables.
- `prepare_manifest.py` now automatically excludes paths listed in `ml/data/eval/holdout_exclude_paths.txt`, so fixed holdout examples do not leak back into training.

## Storage lifecycle policy (implemented)
- Inbox batches are working space under `ml/data/inbox/<batch>`.
- `Finalize Batch (Auto)` now:
  - promotes labeled clips (with hash dedup),
  - retrains model,
  - cleans heavy batch artifacts (`raw_long`, `clips`, score CSVs).
- Retention cleanup:
  - Use `Run Retention Cleanup` in dashboard,
  - default TTL is `14` days for inbox cleanup,
  - optional hard-delete of whole old batch folders.

## Notes
- Baseline model is logistic regression on handcrafted audio features.
- `train_v2.py` is the current stronger candidate: richer audio features + ExtraTrees + isotonic calibration.
- Baseline is still useful as a regression check, but it is not the recommended final route.
- True end-state should still move beyond handcrafted features toward a pretrained audio model / AST-style fine-tuning.
- Baseline training now defaults to grouped splitting by source-like key to reduce clip leakage.
- Model package stores decision threshold and Platt calibration parameters; scoring uses calibrated probability.
