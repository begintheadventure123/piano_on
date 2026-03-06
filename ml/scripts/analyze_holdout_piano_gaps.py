import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


def compute_descriptors(path: str, sample_rate: int = 16000) -> dict[str, float]:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms_mean": 0.0,
            "rms_p90": 0.0,
            "onset_mean": 0.0,
            "onset_p90": 0.0,
            "onset_peak_count": 0.0,
            "centroid_mean": 0.0,
            "rolloff_mean": 0.0,
            "flatness_mean": 0.0,
            "zcr_mean": 0.0,
            "chroma_peak_mean": 0.0,
            "harmonic_ratio": 0.0,
        }

    eps = 1e-10
    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    harmonic, percussive = librosa.effects.hpss(y)

    onset_thresh = float(np.percentile(onset, 90)) if onset.size else 0.0
    onset_peak_count = float(np.sum(onset >= onset_thresh)) if onset.size else 0.0
    harmonic_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + eps))

    return {
        "duration_seconds": float(len(y) / max(sr, 1)),
        "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
        "rms_p90": float(np.percentile(rms, 90)) if rms.size else 0.0,
        "onset_mean": float(np.mean(onset)) if onset.size else 0.0,
        "onset_p90": float(np.percentile(onset, 90)) if onset.size else 0.0,
        "onset_peak_count": onset_peak_count,
        "centroid_mean": float(np.mean(centroid)) if centroid.size else 0.0,
        "rolloff_mean": float(np.mean(rolloff)) if rolloff.size else 0.0,
        "flatness_mean": float(np.mean(flatness)) if flatness.size else 0.0,
        "zcr_mean": float(np.mean(zcr)) if zcr.size else 0.0,
        "chroma_peak_mean": float(np.mean(np.max(chroma, axis=0))) if chroma.size else 0.0,
        "harmonic_ratio": harmonic_ratio,
    }


def effect_size(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare holdout piano false negatives vs true positives")
    parser.add_argument("--preds", default="ml/reports/piano_detector_v23_ensemble_holdout_predictions.csv")
    parser.add_argument("--holdout-index", default="ml/data/eval/holdout/holdout_index.csv")
    parser.add_argument("--out-csv", default="ml/reports/piano_detector_v23_ensemble_piano_gap_features.csv")
    parser.add_argument("--out-json", default="ml/reports/piano_detector_v23_ensemble_piano_gap_summary.json")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    preds = pd.read_csv(args.preds)
    holdout_index = pd.read_csv(args.holdout_index)
    holdout_index = holdout_index[holdout_index["human_label"].astype(str).str.lower() == "piano"].copy()
    holdout_index["holdout_path_norm"] = holdout_index["holdout_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    preds["holdout_path_norm"] = preds["path"].astype(str).str.replace("/", "\\", regex=False).str.lower()

    merged = preds.merge(
        holdout_index[["holdout_path_norm", "dataset_id", "resolved_source_path"]],
        on="holdout_path_norm",
        how="left",
    )
    piano = merged[merged["label"] == 1].copy()
    piano["bucket"] = np.where(piano["prediction"] == 0, "false_negative", "true_positive")

    rows = []
    for row in piano.itertuples(index=False):
        audio_path = str(getattr(row, "resolved_source_path", "") or "").strip()
        if not audio_path:
            continue
        desc = compute_descriptors(audio_path, sample_rate=int(args.sample_rate))
        desc.update(
            {
                "holdout_path": str(row.path),
                "dataset_id": str(getattr(row, "dataset_id", "") or ""),
                "resolved_source_path": audio_path,
                "bucket": str(getattr(row, "bucket")),
                "ensemble_probability": float(getattr(row, "piano_probability")),
            }
        )
        rows.append(desc)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    features = [
        "rms_mean",
        "rms_p90",
        "onset_mean",
        "onset_p90",
        "onset_peak_count",
        "centroid_mean",
        "rolloff_mean",
        "flatness_mean",
        "zcr_mean",
        "chroma_peak_mean",
        "harmonic_ratio",
    ]
    fn = df[df["bucket"] == "false_negative"].copy()
    tp = df[df["bucket"] == "true_positive"].copy()
    comparisons = []
    for feature in features:
        comparisons.append(
            {
                "feature": feature,
                "false_negative_mean": float(fn[feature].mean()),
                "true_positive_mean": float(tp[feature].mean()),
                "false_negative_median": float(fn[feature].median()),
                "true_positive_median": float(tp[feature].median()),
                "effect_size_fn_minus_tp": effect_size(fn[feature], tp[feature]),
            }
        )
    comparisons.sort(key=lambda row: abs(row["effect_size_fn_minus_tp"]), reverse=True)

    summary = {
        "false_negative_count": int(len(fn)),
        "true_positive_count": int(len(tp)),
        "false_negative_by_dataset": fn["dataset_id"].value_counts().to_dict(),
        "top_feature_differences": comparisons[:8],
    }
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"feature_csv={out_csv.resolve()}")
    print(f"summary_json={out_json.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
