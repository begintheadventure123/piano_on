import argparse
from pathlib import Path

import pandas as pd


def parse_weights(raw: str, n: int) -> list[float]:
    if not raw:
        return [1.0 / n] * n
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise RuntimeError(f"weights count ({len(vals)}) must equal number of score files ({n})")
    s = sum(vals)
    if s <= 0:
        raise RuntimeError("weights sum must be positive")
    return [v / s for v in vals]


def load_score_csv(path: Path, idx: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"path", "piano_probability"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{path} missing columns: {miss}")
    out = df[["path", "piano_probability"]].copy()
    out = out.rename(columns={"piano_probability": f"p_{idx}"})
    return out


def main():
    parser = argparse.ArgumentParser(description="Combine probability outputs from multiple models")
    parser.add_argument("--scores", required=True, help="Comma-separated score CSV paths")
    parser.add_argument("--weights", default="", help="Comma-separated weights aligned with --scores")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    score_paths = [Path(x.strip()) for x in args.scores.split(",") if x.strip()]
    if not score_paths:
        raise RuntimeError("No score files provided")
    for p in score_paths:
        if not p.exists():
            raise FileNotFoundError(f"Score file not found: {p}")

    weights = parse_weights(args.weights, len(score_paths))
    merged = None
    for i, p in enumerate(score_paths):
        d = load_score_csv(p, i)
        merged = d if merged is None else merged.merge(d, on="path", how="inner")
    if merged is None or merged.empty:
        raise RuntimeError("No overlapping paths among model score files")

    prob_cols = [f"p_{i}" for i in range(len(score_paths))]
    for c in prob_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna(subset=prob_cols).copy()
    merged["piano_probability"] = 0.0
    for c, w in zip(prob_cols, weights):
        merged["piano_probability"] += merged[c] * w
    merged["predicted_label"] = merged["piano_probability"].map(lambda p: "piano" if p >= args.threshold else "non_piano")
    merged["decision_threshold"] = float(args.threshold)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"rows={len(merged)}")
    print(f"weights={weights}")
    print(f"threshold={args.threshold:.4f}")
    print(f"out_csv={out}")


if __name__ == "__main__":
    main()
