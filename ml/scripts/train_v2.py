import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from model_utils import (
    build_metrics,
    choose_threshold_balanced_accuracy,
    choose_threshold_for_recall,
    grouped_split_indices,
    source_group_key,
)
from model_utils_v2 import featurize_v2


ROOT = Path(__file__).resolve().parents[2]


def load_train_weight_map(path: Path, cfg: dict | None = None) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    boost_cfg = dict((cfg or {}).get("hard_positive_boosts", {}))
    non_piano_weight = float(boost_cfg.get("non_piano_weight", 3.0))
    review_copy_piano_weight = float(boost_cfg.get("review_copy_piano_weight", 4.0))
    review_copy_non_piano_weight = float(boost_cfg.get("review_copy_non_piano_weight", 3.0))
    home_piano_dataset_weight = float(boost_cfg.get("home_piano_dataset_weight", 8.0))
    home_piano_path_weight = float(boost_cfg.get("home_piano_path_weight", 8.0))
    low_confidence_home_path_weight = float(boost_cfg.get("low_confidence_home_path_weight", 10.0))
    out = {}
    for row in df.itertuples(index=False):
        managed_path = str(getattr(row, "canonical_source_path", "") or "").strip()
        label = str(getattr(row, "human_label", "") or "").strip().lower()
        source_kind = str(getattr(row, "source_kind", "") or "").strip().lower()
        dataset_id = str(getattr(row, "dataset_id", "") or "").strip().lower()
        path_lower = managed_path.lower()
        weight = 1.0
        if label == "non_piano":
            weight = max(weight, non_piano_weight)
        if source_kind == "review_copy":
            weight = max(weight, review_copy_piano_weight if label == "piano" else review_copy_non_piano_weight)
        if "home_piano_review_top60" in dataset_id and label == "piano":
            weight = max(weight, home_piano_dataset_weight)
        if label == "piano" and "home_piano" in path_lower:
            weight = max(weight, home_piano_path_weight)
        if label == "piano" and "home_piano_low_confidence" in path_lower:
            weight = max(weight, low_confidence_home_path_weight)
        if managed_path:
            out[str(Path(managed_path).resolve())] = float(weight)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 piano/non-piano classifier with richer audio features")
    parser.add_argument("--manifest", default="ml/data/labels/manifest.csv")
    parser.add_argument("--config", default="ml/configs/train_config_v2.yaml")
    parser.add_argument("--model-out", default="ml/models/piano_detector_v2.joblib")
    parser.add_argument("--report-out", default="ml/reports/piano_detector_v2_metrics.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.manifest)
    weight_map = load_train_weight_map(ROOT / "ml" / "data" / "review" / "asset_pool" / "train_index.csv", cfg=cfg)

    X = []
    y = []
    groups = []
    kept_paths = []
    sample_weights = []
    for row in df.itertuples(index=False):
        try:
            resolved_path = str(Path(row.path).resolve())
            X.append(featurize_v2(row.path, cfg))
            y.append(int(row.label))
            groups.append(source_group_key(row.path))
            kept_paths.append(resolved_path)
            sample_weights.append(float(weight_map.get(resolved_path, 1.0)))
        except Exception as ex:
            print(f"skip {row.path}: {ex}")

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    sample_weights = np.array(sample_weights, dtype=np.float64)
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
            X_train, X_val, X_test = X[tr_idx], X[va_idx], X[te_idx]
            y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                raise RuntimeError("Grouped split produced a degenerate class distribution")
        except Exception as ex:
            print(f"group split fallback to stratified split: {ex}")
            split_mode = "stratified_fallback"
            tr_idx, temp_idx, y_train, y_temp = train_test_split(
                all_idx,
                y,
                test_size=(1.0 - float(cfg["train_split"])),
                random_state=int(cfg["seed"]),
                stratify=y,
            )
            val_ratio = float(cfg["val_split"]) / (float(cfg["val_split"]) + float(cfg["test_split"]))
            va_idx, te_idx, y_val, y_test = train_test_split(
                temp_idx,
                y_temp,
                test_size=(1.0 - val_ratio),
                random_state=int(cfg["seed"]),
                stratify=y_temp,
            )
            X_train, X_val, X_test = X[tr_idx], X[va_idx], X[te_idx]
    else:
        split_mode = "stratified_config_disabled"
        tr_idx, temp_idx, y_train, y_temp = train_test_split(
            all_idx,
            y,
            test_size=(1.0 - float(cfg["train_split"])),
            random_state=int(cfg["seed"]),
            stratify=y,
        )
        val_ratio = float(cfg["val_split"]) / (float(cfg["val_split"]) + float(cfg["test_split"]))
        va_idx, te_idx, y_val, y_test = train_test_split(
            temp_idx,
            y_temp,
            test_size=(1.0 - val_ratio),
            random_state=int(cfg["seed"]),
            stratify=y_temp,
        )
        X_train, X_val, X_test = X[tr_idx], X[va_idx], X[te_idx]

    model = ExtraTreesClassifier(
        n_estimators=int(cfg["classifier"].get("n_estimators", 700)),
        min_samples_leaf=int(cfg["classifier"].get("min_samples_leaf", 2)),
        max_features=str(cfg["classifier"].get("max_features", "sqrt")),
        class_weight=str(cfg["classifier"].get("class_weight", "balanced_subsample")),
        random_state=int(cfg["seed"]),
        n_jobs=int(cfg["classifier"].get("n_jobs", 1)),
    )
    train_weights = sample_weights[tr_idx]
    model.fit(X_train, y_train, sample_weight=train_weights)

    val_prob_raw = model.predict_proba(X_val)[:, 1]
    test_prob_raw = model.predict_proba(X_test)[:, 1]

    calibration_cfg = cfg.get("calibration", {})
    calibration_enabled = bool(calibration_cfg.get("enabled", True))
    calibration_method = str(calibration_cfg.get("method", "isotonic")).strip().lower()
    calibrator = None
    min_calibration_samples = int(calibration_cfg.get("min_val_samples", 80))
    min_calibration_per_class = int(calibration_cfg.get("min_val_per_class", 20))
    val_class_counts = pd.Series(y_val).value_counts().to_dict()
    val_too_small = (
        len(y_val) < min_calibration_samples
        or int(val_class_counts.get(0, 0)) < min_calibration_per_class
        or int(val_class_counts.get(1, 0)) < min_calibration_per_class
    )
    if calibration_enabled and calibration_method == "isotonic" and not val_too_small:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_prob_raw, y_val)
        val_prob = calibrator.predict(val_prob_raw)
        test_prob = calibrator.predict(test_prob_raw)
    else:
        calibration_enabled = False
        calibration_method = "none"
        val_prob = val_prob_raw
        test_prob = test_prob_raw

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
    val_metrics = build_metrics(y_val, val_prob, decision_threshold, recall_target)
    test_metrics = build_metrics(y_test, test_prob, decision_threshold, recall_target)

    metrics = {
        "model_family": "extra_trees_v2",
        "feature_mode": "rich_v2",
        "num_samples": int(len(y)),
        "split_mode": split_mode,
        "split_counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "decision_threshold": float(decision_threshold),
        "review_threshold": float(review_threshold),
        "calibration": {
            "enabled": calibration_enabled,
            "method": calibration_method,
        },
        "thresholding": {
            "policy": threshold_policy,
            "review_recall_target": review_recall_target,
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
            "feature_mode": "rich_v2",
            "model_family": "extra_trees_v2",
            "decision_threshold": float(decision_threshold),
            "review_threshold": float(review_threshold),
            "calibration": {
                "enabled": calibration_enabled,
                "method": calibration_method,
                "model": calibrator,
            },
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
