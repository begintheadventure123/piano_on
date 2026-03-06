# ML Architecture

## Current Source Of Truth

- Review queues: `ml/data/ops/activity.db` tables `review_batches`, `review_items`
- Human review labels: `ml/data/ops/activity.db` tables `review_labels`, `review_events`
- Promoted training assets: `ml/data/raw/piano/hard_positive`, `ml/data/raw/non_piano/hard_negative`
- Fixed evaluation set: `ml/data/eval/holdout`

## Primary Entrypoints

- Dashboard: `ml/tools/clip_reviewer/server.py`
- Batch loop: `ml/scripts/iterate_training.py`
- Manifest build: `ml/scripts/prepare_manifest.py`
- Model train: `ml/scripts/train_baseline.py`
- Holdout eval: `ml/scripts/evaluate_holdout.py`

## Workflow

1. Upload long recordings into `ml/data/inbox/<batch>/raw_long`
2. Run `iterate_training.py run-batch`
3. Review clips in dashboard; labels auto-save to DB
4. Run `promote-labels --batch <batch>` or use dashboard finalize
5. Rebuild manifest and retrain
6. Check holdout before trusting a new model

## Legacy Artifacts

- `false_positive_review.csv`: audit/export artifact only, not the main review store
- `priority_label_queue.csv`: batch report artifact only
- `*_labeled.csv`: legacy export format, no longer required for promote/finalize
- `*_labels.json`: legacy compatibility artifact, no longer part of the primary workflow

## Guardrails

- Manual review history should be archived with `ml/scripts/archive_manual_labels.py`
- Holdout samples listed in `ml/data/eval/holdout_exclude_paths.txt` must never re-enter training
- Treat `activity.db` as the authoritative review history
