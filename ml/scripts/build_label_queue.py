import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build priority labeling queue from clip scores")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--high-prob-min", type=float, default=0.9)
    parser.add_argument("--top-hard", type=int, default=300)
    parser.add_argument("--top-uncertain", type=int, default=200)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)
    needed = {"path", "piano_probability", "predicted_label"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise RuntimeError(f"Scores CSV missing required columns: {miss}")

    work = df.dropna(subset=["piano_probability"]).copy()
    work["piano_probability"] = pd.to_numeric(work["piano_probability"], errors="coerce")
    work = work.dropna(subset=["piano_probability"])
    work["distance_to_threshold"] = (work["piano_probability"] - args.decision_threshold).abs()

    hard = work[(work["predicted_label"] == "piano") & (work["piano_probability"] >= args.high_prob_min)].copy()
    hard = hard.sort_values("piano_probability", ascending=False).head(max(0, args.top_hard))
    hard["priority_reason"] = "hard_negative_candidate_high_prob_piano"

    uncertain = work.sort_values("distance_to_threshold", ascending=True).head(max(0, args.top_uncertain)).copy()
    uncertain["priority_reason"] = "near_decision_boundary"

    merged = (
        pd.concat([hard, uncertain], ignore_index=True)
        .drop_duplicates(subset=["path"], keep="first")
        .sort_values(["priority_reason", "piano_probability"], ascending=[True, False])
        .reset_index(drop=True)
    )
    merged.insert(0, "queue_rank", range(1, len(merged) + 1))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"queue_size={len(merged)}")
    print(f"hard_candidates={len(hard)}")
    print(f"uncertain_candidates={len(uncertain)}")
    print(f"out_csv={out}")


if __name__ == "__main__":
    main()
