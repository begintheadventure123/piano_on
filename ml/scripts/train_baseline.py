import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib


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
    parser = argparse.ArgumentParser(description="Train baseline piano/non-piano classifier")
    parser.add_argument("--manifest", default="ml/data/labels/manifest.csv")
    parser.add_argument("--config", default="ml/configs/train_config.yaml")
    parser.add_argument("--model-out", default="ml/models/baseline_logreg.joblib")
    parser.add_argument("--report-out", default="ml/reports/baseline_metrics.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.manifest)

    X = []
    y = []
    kept_paths = []
    for row in df.itertuples(index=False):
        try:
            X.append(featurize(row.path, cfg))
            y.append(int(row.label))
            kept_paths.append(row.path)
        except Exception as ex:
            print(f"skip {row.path}: {ex}")

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)

    unique = np.unique(y)
    if unique.size < 2:
        raise RuntimeError(
            "Training requires at least 2 classes. Add non-piano clips under ml/data/raw/non_piano."
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - cfg["train_split"]), random_state=cfg["seed"], stratify=y
    )

    val_ratio = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=cfg["seed"], stratify=y_temp
    )

    c = cfg["classifier"]["C"]
    max_iter = cfg["classifier"]["max_iter"]
    class_weight = cfg["classifier"]["class_weight"]

    model = LogisticRegression(C=c, max_iter=max_iter, class_weight=class_weight)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "val_report": classification_report(y_val, val_pred, output_dict=True),
        "test_report": classification_report(y_test, test_pred, output_dict=True),
        "val_auc": float(roc_auc_score(y_val, val_prob)),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "num_samples": int(len(y)),
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": cfg}, model_out)

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"model: {model_out}")
    print(f"report: {report_out}")
    print(f"val_auc={metrics['val_auc']:.4f} test_auc={metrics['test_auc']:.4f}")


if __name__ == "__main__":
    main()
