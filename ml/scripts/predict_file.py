import argparse
from pathlib import Path
import numpy as np
import librosa
import joblib
import yaml


def featurize(path: str, cfg: dict) -> np.ndarray:
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=True, duration=cfg["max_duration_seconds"])
    if y.size == 0:
        return np.zeros(12, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=2.0,
    )
    logmel = librosa.power_to_db(mel + 1e-10)
    feats = [
        float(np.mean(logmel)),
        float(np.std(logmel)),
        float(np.percentile(logmel, 10)),
        float(np.percentile(logmel, 50)),
        float(np.percentile(logmel, 90)),
    ]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    for arr in (centroid, rolloff, zcr, rms):
        feats.append(float(np.mean(arr)))
        feats.append(float(np.std(arr)))
    return np.array(feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run baseline model on one audio file")
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="ml/models/baseline_logreg.joblib")
    parser.add_argument("--config", default="ml/configs/train_config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pack = joblib.load(args.model)
    model = pack["model"]

    x = featurize(args.audio_path, cfg).reshape(1, -1)
    p = float(model.predict_proba(x)[0, 1])
    pred = int(p >= 0.5)
    print(f"file={args.audio_path}")
    print(f"piano_probability={p:.4f}")
    print(f"prediction={'piano' if pred else 'non_piano'}")


if __name__ == "__main__":
    main()
