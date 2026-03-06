import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from model_utils import build_metrics, choose_threshold_balanced_accuracy, choose_threshold_for_recall, grouped_split_indices, source_group_key
from model_utils_v3 import featurize_v3
from train_v2 import load_train_weight_map


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v3 temporal piano/non-piano classifier")
    parser.add_argument("--manifest", default="ml/data/labels/manifest.csv")
    parser.add_argument("--config", default="ml/configs/train_config_v3.yaml")
    parser.add_argument("--model-out", default="ml/models/piano_detector_v3.joblib")
    parser.add_argument("--report-out", default="ml/reports/piano_detector_v3_metrics.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.manifest)
    weight_map = load_train_weight_map(ROOT / "ml" / "data" / "review" / "asset_pool" / "train_index.csv")

    X = []
    y = []
    groups = []
    sample_weights = []
    for row in df.itertuples(index=False):
        try:
            resolved = str(Path(row.path).resolve())
            X.append(featurize_v3(row.path, cfg))
            y.append(int(row.label))
            groups.append(source_group_key(row.path))
            sample_weights.append(float(weight_map.get(resolved, 1.0)))
        except Exception as ex:
            print(f"skip {row.path}: {ex}")

    X = np.vstack(X)
    y = np.asarray(y, dtype=np.int64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if np.unique(y).size < 2:
        raise RuntimeError("Training requires at least 2 classes")

    use_group_split = bool(cfg.get("group_split", {}).get("enabled", True))
    split_mode = "grouped"
    all_idx = np.arange(len(y))
    if use_group_split:
        try:
            tr_idx, va_idx, te_idx = grouped_split_indices(
                y=y,
                groups=groups,
                train_size=float(cfg["train_split"]),
                val_size=float(cfg["val_split"]),
                test_size=float(cfg["test_split"]),
                seed=int(cfg["seed"]),
            )
            if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
                raise RuntimeError("Grouped split produced a degenerate class distribution")
        except Exception as ex:
            print(f"group split fallback to stratified split: {ex}")
            split_mode = "stratified_fallback"
            tr_idx, temp_idx, _, y_temp = train_test_split(
                all_idx,
                y,
                test_size=(1.0 - float(cfg["train_split"])),
                random_state=int(cfg["seed"]),
                stratify=y,
            )
            val_ratio = float(cfg["val_split"]) / (float(cfg["val_split"]) + float(cfg["test_split"]))
            va_idx, te_idx, _, _ = train_test_split(
                temp_idx,
                y_temp,
                test_size=(1.0 - val_ratio),
                random_state=int(cfg["seed"]),
                stratify=y_temp,
            )
    else:
        split_mode = "stratified_config_disabled"
        tr_idx, temp_idx, _, y_temp = train_test_split(
            all_idx,
            y,
            test_size=(1.0 - float(cfg["train_split"])),
            random_state=int(cfg["seed"]),
            stratify=y,
        )
        val_ratio = float(cfg["val_split"]) / (float(cfg["val_split"]) + float(cfg["test_split"]))
        va_idx, te_idx, _, _ = train_test_split(
            temp_idx,
            y_temp,
            test_size=(1.0 - val_ratio),
            random_state=int(cfg["seed"]),
            stratify=y_temp,
        )

    X_train, X_val, X_test = X[tr_idx], X[va_idx], X[te_idx]
    y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]

    clf_cfg = cfg.get("classifier", {})
    model = HistGradientBoostingClassifier(
        learning_rate=float(clf_cfg.get("learning_rate", 0.05)),
        max_iter=int(clf_cfg.get("max_iter", 350)),
        max_leaf_nodes=int(clf_cfg.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(clf_cfg.get("min_samples_leaf", 20)),
        l2_regularization=float(clf_cfg.get("l2_regularization", 0.0)),
        random_state=int(cfg["seed"]),
    )
    model.fit(X_train, y_train, sample_weight=sample_weights[tr_idx])

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    recall_target = float(cfg.get("metrics", {}).get("recall_target_for_fpr", 0.95))
    threshold_cfg = cfg.get("thresholding", {})
    threshold_policy = str(threshold_cfg.get("policy", "balanced_accuracy")).strip().lower()
    if threshold_policy == "recall_target":
        decision_threshold = choose_threshold_for_recall(y_val, val_prob, recall_target, default=0.5)
    else:
        decision_threshold = choose_threshold_balanced_accuracy(
            y_val,
            val_prob,
            default=float(threshold_cfg.get("default", 0.5)),
            min_threshold=float(threshold_cfg.get("min_threshold", 0.05)),
            max_threshold=float(threshold_cfg.get("max_threshold", 0.95)),
        )
    review_recall_target = float(threshold_cfg.get("review_recall_target", 0.98))
    review_threshold = choose_threshold_for_recall(
        y_val,
        val_prob,
        review_recall_target,
        default=max(0.05, min(0.95, float(decision_threshold))),
    )

    metrics = {
        "model_family": "hist_gbdt_temporal_v3",
        "feature_mode": "temporal_v3",
        "num_samples": int(len(y)),
        "split_mode": split_mode,
        "split_counts": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "decision_threshold": float(decision_threshold),
        "review_threshold": float(review_threshold),
        "thresholding": {"policy": threshold_policy, "review_recall_target": review_recall_target},
        "val": build_metrics(y_val, val_prob, decision_threshold, recall_target),
        "test": build_metrics(y_test, test_prob, decision_threshold, recall_target),
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "config": cfg,
            "feature_mode": "temporal_v3",
            "model_family": "hist_gbdt_temporal_v3",
            "decision_threshold": float(decision_threshold),
            "review_threshold": float(review_threshold),
            "calibration": {"enabled": False, "method": "none", "model": None},
            "training_meta": {
                "split_mode": split_mode,
                "num_samples": int(len(y)),
                "num_features": int(X.shape[1]),
                "weighted_examples": int(np.sum(sample_weights > 1.0)),
            },
        },
        model_out,
    )

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"model: {model_out}")
    print(f"report: {report_out}")
    print(f"threshold={metrics['decision_threshold']:.4f}")
    print(f"val_auc={metrics['val']['auc_roc']:.4f} test_auc={metrics['test']['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
