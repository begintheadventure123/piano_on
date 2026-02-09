# Training Pipeline Setup Status

Done. I set up the training pipeline foundation and downloaded starter piano samples.

## What I set up
1. ML project structure under `ml/`:
- `ml/configs`
- `ml/scripts`
- `ml/data/raw/{piano,non_piano,mixed}`
- `ml/data/{interim,processed,labels}`
- `ml/models`, `ml/reports`, `ml/logs`

2. Python/venv framework:
- Installed Python `3.11` (side-by-side)
- Created venv: `.venv-ml`
- Installed requirements from `ml/requirements.txt` (numpy/pandas/sklearn/librosa/etc.)
- Added setup script: `ml/scripts/setup_env.ps1`

3. Baseline training pipeline scripts:
- `ml/scripts/download_uiowa_piano.py`
- `ml/scripts/prepare_manifest.py`
- `ml/scripts/train_baseline.py`
- `ml/scripts/predict_file.py`
- Config: `ml/configs/train_config.yaml`
- Guide: `ml/README.md`

4. Downloaded online piano samples:
- Downloaded `80` files into `ml/data/raw/piano/uiowa`

Source used:
- University of Iowa instrument samples page: `https://theremin.music.uiowa.edu/MISpiano.html`
- Example direct sample URL: `https://theremin.music.uiowa.edu/sound%20files/MIS%20Piano/Piano.ff.Bb0.aiff`

## Important current status
- Training is not runnable yet because only positive data exists right now.
- `train_baseline.py` correctly stops with:
  - "Add non-piano clips under `ml/data/raw/non_piano`".

## What you need to do next (exactly)
1. Add negative clips:
- Put at least `50+` non-piano clips into `ml/data/raw/non_piano`
- Examples: speech, TV, street noise, typing, other instruments.

2. (Optional) Add your mixed clips:
- Put into `ml/data/raw/mixed`

3. Run pipeline:
```powershell
.\.venv-ml\Scripts\Activate.ps1
python ml/scripts/prepare_manifest.py
python ml/scripts/train_baseline.py
```

4. Test one file:
```powershell
python ml/scripts/predict_file.py path\to\your\file.m4a
```

If you want, next I can add:
1. automatic segment-level labeling format (`labels.json`)
2. threshold sweep + precision/recall report
3. AST-embedding training path (stronger than the current handcrafted baseline).
