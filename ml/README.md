# ML Training Pipeline

This folder contains the offline training/evaluation pipeline for piano activity detection.

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
4. Add your own non-piano clips under `ml/data/raw/non_piano`.
5. Build manifest:
   - `python ml/scripts/prepare_manifest.py`
6. Train baseline:
   - `python ml/scripts/train_baseline.py`
7. Predict one file:
   - `python ml/scripts/predict_file.py path\\to\\audio.m4a`

## Notes
- Baseline model is logistic regression on handcrafted audio features.
- This is a calibration baseline before moving to AST fine-tuning.
