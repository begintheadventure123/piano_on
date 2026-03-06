import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inference_utils import decision_threshold_from_pack, load_effective_config, load_model_pack, predict_probability
from model_utils import build_metrics

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".wma", ".flac", ".aiff", ".aif"}


def iter_audio(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def collect_holdout(holdout_root: Path) -> pd.DataFrame:
    rows = []
    for sub, label in [("non_piano", 0), ("piano", 1)]:
        base = holdout_root / sub
        if not base.exists():
            continue
        for p in iter_audio(base):
            rows.append({"path": str(p), "label": int(label)})
    if not rows:
        raise RuntimeError(f"No holdout audio found under {holdout_root}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on fixed holdout set")
    parser.add_argument("--holdout-root", default="ml/data/eval/holdout")
    parser.add_argument("--model", default="ml/models/baseline_logreg.joblib")
    parser.add_argument("--config", default="ml/configs/train_config.yaml")
    parser.add_argument("--out-report", default="ml/reports/holdout_metrics.json")
    parser.add_argument("--out-preds", default="ml/reports/holdout_predictions.csv")
    args = parser.parse_args()

    holdout_root = Path(args.holdout_root)
    df = collect_holdout(holdout_root)
    pack = load_model_pack(args.model)
    cfg = load_effective_config(pack, args.config)
    threshold = decision_threshold_from_pack(pack)
    recall_target = float(cfg.get("metrics", {}).get("recall_target_for_fpr", 0.95))

    probs_raw = []
    probs = []
    preds = []
    labels = []
    paths = []
    errors = 0
    for row in df.itertuples(index=False):
        try:
            p_raw, p = predict_probability(row.path, pack, cfg)
            probs_raw.append(p_raw)
            probs.append(p)
            preds.append(int(p >= threshold))
            labels.append(int(row.label))
            paths.append(str(row.path))
        except Exception:
            errors += 1

    if not labels:
        raise RuntimeError("No holdout sample could be scored")

    y_true = np.array(labels, dtype=np.int64)
    y_prob = np.array(probs, dtype=np.float64)
    metrics = build_metrics(y_true, y_prob, threshold, recall_target)
    metrics["num_samples"] = int(len(labels))
    metrics["num_errors"] = int(errors)
    metrics["threshold"] = float(threshold)
    metrics["model"] = str(Path(args.model).resolve())
    metrics["holdout_root"] = str(holdout_root.resolve())

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    out_preds = Path(args.out_preds)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "path": paths,
            "label": labels,
            "prediction": preds,
            "piano_probability_raw": probs_raw,
            "piano_probability": probs,
        }
    ).to_csv(out_preds, index=False)

    print(f"holdout_samples={len(labels)}")
    print(f"errors={errors}")
    print(f"threshold={threshold:.4f}")
    print(f"auc_roc={metrics['auc_roc']:.4f}")
    print(f"report={out_report}")
    print(f"predictions={out_preds}")


if __name__ == "__main__":
    main()
