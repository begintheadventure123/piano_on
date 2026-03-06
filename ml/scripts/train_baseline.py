import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from model_utils import (
    apply_platt,
    build_metrics,
    choose_threshold_for_recall,
    featurize,
    grouped_split_indices,
    source_group_key,
    stratified_fallback_split,
)


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
    groups = []
    kept_paths = []
    for row in df.itertuples(index=False):
        try:
            X.append(featurize(row.path, cfg))
            y.append(int(row.label))
            groups.append(source_group_key(row.path))
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

    use_group_split = bool(cfg.get("group_split", {}).get("enabled", True))
    split_mode = "grouped"
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
            X_train, X_val, X_test = X[tr_idx], X[va_idx], X[te_idx]
            y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                raise RuntimeError("Grouped split produced a degenerate class distribution")
        except Exception as ex:
            print(f"group split fallback to stratified split: {ex}")
            split_mode = "stratified_fallback"
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_fallback_split(
                X,
                y,
                train_size=float(cfg["train_split"]),
                val_size=float(cfg["val_split"]),
                test_size=float(cfg["test_split"]),
                seed=int(cfg["seed"]),
            )
    else:
        split_mode = "stratified_config_disabled"
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_fallback_split(
            X,
            y,
            train_size=float(cfg["train_split"]),
            val_size=float(cfg["val_split"]),
            test_size=float(cfg["test_split"]),
            seed=int(cfg["seed"]),
        )

    c = cfg["classifier"]["C"]
    max_iter = cfg["classifier"]["max_iter"]
    class_weight = cfg["classifier"]["class_weight"]

    # Standardizing handcrafted features improves optimizer stability for logistic regression.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=max_iter, class_weight=class_weight),
    )
    model.fit(X_train, y_train)

    val_prob_raw = model.predict_proba(X_val)[:, 1]
    test_prob_raw = model.predict_proba(X_test)[:, 1]

    calibration_cfg = cfg.get("calibration", {})
    calibration_enabled = bool(calibration_cfg.get("enabled", True))
    calibration_method = str(calibration_cfg.get("method", "platt")).lower()
    platt_coef = 1.0
    platt_intercept = 0.0
    if calibration_enabled and calibration_method == "platt":
        calib = LogisticRegression(C=1.0, max_iter=1000, class_weight=None)
        calib.fit(val_prob_raw.reshape(-1, 1), y_val)
        platt_coef = float(calib.coef_[0, 0])
        platt_intercept = float(calib.intercept_[0])
        val_prob = apply_platt(val_prob_raw, platt_coef, platt_intercept)
        test_prob = apply_platt(test_prob_raw, platt_coef, platt_intercept)
    else:
        calibration_enabled = False
        val_prob = val_prob_raw
        test_prob = test_prob_raw

    recall_target = float(cfg.get("metrics", {}).get("recall_target_for_fpr", 0.95))
    decision_threshold = choose_threshold_for_recall(y_val, val_prob, recall_target, default=0.5)
    val_metrics = build_metrics(y_val, val_prob, decision_threshold, recall_target)
    test_metrics = build_metrics(y_test, test_prob, decision_threshold, recall_target)
    metrics = {
        "num_samples": int(len(y)),
        "split_mode": split_mode,
        "split_counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "decision_threshold": float(decision_threshold),
        "calibration": {
            "enabled": calibration_enabled,
            "method": "platt" if calibration_enabled else "none",
            "coef": float(platt_coef),
            "intercept": float(platt_intercept),
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "config": cfg,
            "decision_threshold": float(decision_threshold),
            "calibration": {
                "enabled": calibration_enabled,
                "method": "platt" if calibration_enabled else "none",
                "coef": float(platt_coef),
                "intercept": float(platt_intercept),
            },
            "training_meta": {
                "split_mode": split_mode,
                "num_samples": int(len(y)),
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
