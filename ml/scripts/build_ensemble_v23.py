import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted ensemble from v2 and v3 piano models")
    parser.add_argument("--v2-model", default="ml/models/piano_detector_v2.joblib")
    parser.add_argument("--v2-config", default="ml/configs/train_config_v2.yaml")
    parser.add_argument("--v3-model", default="ml/models/piano_detector_v3.joblib")
    parser.add_argument("--v3-config", default="ml/configs/train_config_v3.yaml")
    parser.add_argument("--out-model", default="ml/models/piano_detector_v23_ensemble.joblib")
    parser.add_argument("--out-report", default="ml/reports/piano_detector_v23_ensemble_build.json")
    parser.add_argument("--weight-v2", type=float, default=0.7)
    parser.add_argument("--weight-v3", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.84)
    parser.add_argument("--review-threshold", type=float, default=0.72)
    parser.add_argument("--method", default="weighted_average", choices=["weighted_average", "min", "max"])
    args = parser.parse_args()

    v2_model = Path(args.v2_model).resolve()
    v3_model = Path(args.v3_model).resolve()
    v2_pack = joblib.load(v2_model)
    v3_pack = joblib.load(v3_model)

    weights = [float(args.weight_v2), float(args.weight_v3)]
    total = sum(weights)
    if total <= 0:
        raise RuntimeError("Ensemble weights must sum to a positive value")
    weights = [w / total for w in weights]

    ensemble_pack = {
        "model_family": "ensemble_v23",
        "feature_mode": "ensemble_v1",
        "ensemble_method": str(args.method).strip().lower(),
        "ensemble_weights": weights,
        "decision_threshold": float(args.threshold),
        "review_threshold": float(args.review_threshold),
        "config": {
            "metrics": {"recall_target_for_fpr": 0.95},
            "ensemble": {
                "method": str(args.method).strip().lower(),
                "members": ["v2", "v3"],
                "weights": weights,
            },
        },
        "members": [
            {
                "name": "v2",
                "model_path": str(v2_model),
                "config_path": str(Path(args.v2_config).resolve()),
                "pack": v2_pack,
                "weight": weights[0],
            },
            {
                "name": "v3",
                "model_path": str(v3_model),
                "config_path": str(Path(args.v3_config).resolve()),
                "pack": v3_pack,
                "weight": weights[1],
            },
        ],
        "training_meta": {
            "kind": "built_ensemble",
            "threshold_source": "manual_provisional",
            "notes": "Weighted v2+v3 ensemble built from existing trained packs",
        },
        "calibration": {"enabled": False, "method": "none", "model": None},
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble_pack, out_model)

    build_report = {
        "out_model": str(out_model.resolve()),
        "method": ensemble_pack["ensemble_method"],
        "weights": weights,
        "decision_threshold": float(args.threshold),
        "review_threshold": float(args.review_threshold),
        "members": [
            {"name": "v2", "model": str(v2_model), "decision_threshold": float(v2_pack.get("decision_threshold", 0.5))},
            {"name": "v3", "model": str(v3_model), "decision_threshold": float(v3_pack.get("decision_threshold", 0.5))},
        ],
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(build_report, indent=2), encoding="utf-8")

    print(f"ensemble_model={out_model.resolve()}")
    print(f"build_report={out_report.resolve()}")
    print(f"method={ensemble_pack['ensemble_method']}")
    print(f"weights={weights}")
    print(f"threshold={float(args.threshold):.4f}")


if __name__ == "__main__":
    main()
