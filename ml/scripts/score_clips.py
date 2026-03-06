import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from inference_utils import (
    decision_threshold_from_pack,
    load_effective_config,
    load_model_pack,
    predict_probability,
    review_threshold_from_pack,
)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".wma", ".flac", ".aiff", ".aif"}


def iter_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Score a folder of clips with the baseline piano model")
    parser.add_argument("--input", required=True, help="Input folder of clips")
    parser.add_argument("--model", default="ml/models/baseline_logreg.joblib", help="Trained model file")
    parser.add_argument("--config", default="ml/configs/train_config.yaml", help="Feature config file")
    parser.add_argument("--out-csv", default="ml/reports/clip_scores.csv", help="Full score output CSV")
    parser.add_argument(
        "--review-csv",
        default="ml/reports/clip_low_confidence_review.csv",
        help="Low confidence subset CSV",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.60,
        help="Mark clips with piano probability below this threshold for review",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Always include this many lowest-probability clips in review output",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {in_dir}")

    pack = load_model_pack(args.model)
    cfg = load_effective_config(pack, args.config)
    threshold = decision_threshold_from_pack(pack)
    review_threshold = review_threshold_from_pack(pack)

    rows = []
    for p in iter_audio_files(in_dir):
        try:
            raw_prob, prob = predict_probability(str(p), pack, cfg)
            rows.append(
                {
                    "path": str(p),
                    "piano_probability_raw": raw_prob,
                    "piano_probability": prob,
                    "predicted_label": "piano" if prob >= threshold else "non_piano",
                    "confidence_from_boundary": float(abs(prob - threshold)),
                    "confidence_from_review_threshold": float(abs(prob - review_threshold)),
                }
            )
        except Exception as ex:
            rows.append(
                {
                    "path": str(p),
                    "piano_probability_raw": float("nan"),
                    "piano_probability": np.nan,
                    "predicted_label": "error",
                    "confidence_from_boundary": np.nan,
                    "confidence_from_review_threshold": np.nan,
                    "error": str(ex),
                }
            )

    if not rows:
        raise RuntimeError(f"No audio files found under {in_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["piano_probability", "path"], ascending=[True, True], na_position="first")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    low_df = df[df["piano_probability"] < args.low_threshold]
    top_df = df.head(max(args.top_k, 0))
    review_df = pd.concat([low_df, top_df], ignore_index=True).drop_duplicates(subset=["path"])
    review_df = review_df.sort_values(by=["piano_probability", "path"], ascending=[True, True], na_position="first")

    review_csv = Path(args.review_csv)
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(review_csv, index=False)

    n_error = int((df["predicted_label"] == "error").sum())
    n_review = int(len(review_df))
    n_non_piano_pred = int((df["predicted_label"] == "non_piano").sum())
    print(f"scored={len(df)}")
    print(f"pred_non_piano={n_non_piano_pred}")
    print(f"errors={n_error}")
    print(f"review_clips={n_review}")
    print(f"decision_threshold={threshold:.4f}")
    print(f"review_threshold={review_threshold:.4f}")
    print(f"full_scores={out_csv}")
    print(f"review_scores={review_csv}")


if __name__ == "__main__":
    main()
