import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from estimate_piano_timeline import estimate_piano_timeline, save_timeline_outputs


REQUIRED_COLUMNS = ["audio_path", "start_seconds", "end_seconds", "label"]


def normalize_path_str(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("/", "\\").lower()


def load_annotation_intervals(csv_path: str | Path, audio_path: str | Path, *, positive_labels: set[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Annotation CSV missing required columns {missing}: {csv_path}")

    df = df.copy()
    df["audio_path_norm"] = df["audio_path"].astype(str).map(normalize_path_str)
    target = normalize_path_str(audio_path)
    df = df[df["audio_path_norm"] == target].copy()
    if df.empty:
        raise RuntimeError(f"No annotation rows found for audio file: {audio_path}")

    df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label_norm"].isin(positive_labels)].copy()
    if df.empty:
        return pd.DataFrame(columns=["start_seconds", "end_seconds", "duration_seconds"])

    df["start_seconds"] = pd.to_numeric(df["start_seconds"], errors="coerce")
    df["end_seconds"] = pd.to_numeric(df["end_seconds"], errors="coerce")
    df = df.dropna(subset=["start_seconds", "end_seconds"]).copy()
    df = df[df["end_seconds"] > df["start_seconds"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["start_seconds", "end_seconds", "duration_seconds"])

    df = df.sort_values(by=["start_seconds", "end_seconds"], ascending=[True, True]).reset_index(drop=True)
    merged: list[dict] = []
    for row in df.itertuples(index=False):
        start = float(row.start_seconds)
        end = float(row.end_seconds)
        if merged and start <= float(merged[-1]["end_seconds"]):
            merged[-1]["end_seconds"] = max(float(merged[-1]["end_seconds"]), end)
        else:
            merged.append({"start_seconds": start, "end_seconds": end})
    for idx, item in enumerate(merged, start=1):
        item["interval_index"] = idx
        item["duration_seconds"] = float(item["end_seconds"] - item["start_seconds"])
    return pd.DataFrame(merged)


def sum_duration(df: pd.DataFrame) -> float:
    if df.empty or "duration_seconds" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0.0).sum())


def framewise_mask(intervals_df: pd.DataFrame, audio_duration_seconds: float, resolution_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    if resolution_seconds <= 0:
        raise ValueError("resolution_seconds must be > 0")
    starts = np.arange(0.0, audio_duration_seconds, resolution_seconds, dtype=np.float64)
    ends = np.minimum(starts + resolution_seconds, audio_duration_seconds)
    mask = np.zeros_like(starts, dtype=np.int8)
    if not intervals_df.empty:
        for row in intervals_df.itertuples(index=False):
            start = float(row.start_seconds)
            end = float(row.end_seconds)
            overlap = (ends > start) & (starts < end)
            mask[overlap] = 1
    return starts, mask


def overlap_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame, audio_duration_seconds: float, resolution_seconds: float) -> dict:
    _, pred_mask = framewise_mask(pred_df, audio_duration_seconds, resolution_seconds)
    _, gt_mask = framewise_mask(gt_df, audio_duration_seconds, resolution_seconds)
    intersection = float(np.sum((pred_mask == 1) & (gt_mask == 1)) * resolution_seconds)
    pred_seconds = float(np.sum(pred_mask == 1) * resolution_seconds)
    gt_seconds = float(np.sum(gt_mask == 1) * resolution_seconds)
    union = float(np.sum((pred_mask == 1) | (gt_mask == 1)) * resolution_seconds)
    precision = float(intersection / pred_seconds) if pred_seconds > 0 else 0.0
    recall = float(intersection / gt_seconds) if gt_seconds > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(intersection / union) if union > 0 else 0.0
    return {
        "overlap_seconds": intersection,
        "union_seconds": union,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate piano timeline estimation against manually annotated intervals")
    parser.add_argument("audio_path", help="Audio file to evaluate")
    parser.add_argument("--annotations-csv", required=True, help="CSV with columns audio_path,start_seconds,end_seconds,label")
    parser.add_argument("--positive-labels", default="piano", help="Comma-separated labels treated as piano-positive")
    parser.add_argument("--model", default="ml/models/piano_detector_v23_ensemble.joblib")
    parser.add_argument("--config", default="ml/configs/train_config_v2.yaml")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--smooth-frames", type=int, default=5)
    parser.add_argument("--enter-threshold", type=float, default=None)
    parser.add_argument("--exit-threshold", type=float, default=None)
    parser.add_argument("--min-interval-seconds", type=float, default=1.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--eval-resolution-seconds", type=float, default=0.25)
    parser.add_argument("--out-frames-csv", default="ml/reports/eval_piano_timeline_frames.csv")
    parser.add_argument("--out-intervals-csv", default="ml/reports/eval_piano_timeline_intervals.csv")
    parser.add_argument("--out-summary-json", default="ml/reports/eval_piano_timeline_summary.json")
    parser.add_argument("--out-ground-truth-csv", default="ml/reports/eval_piano_timeline_ground_truth.csv")
    args = parser.parse_args()

    frame_df, pred_intervals_df, summary = estimate_piano_timeline(
        args.audio_path,
        model_path=args.model,
        config_path=args.config,
        window_seconds=float(args.window_seconds),
        hop_seconds=float(args.hop_seconds),
        smooth_frames=int(args.smooth_frames),
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        min_interval_seconds=float(args.min_interval_seconds),
        merge_gap_seconds=float(args.merge_gap_seconds),
    )

    positive_labels = {
        item.strip().lower()
        for item in str(args.positive_labels).split(",")
        if item.strip()
    }
    gt_df = load_annotation_intervals(args.annotations_csv, args.audio_path, positive_labels=positive_labels)

    pred_seconds = float(summary["estimated_piano_seconds"])
    gt_seconds = sum_duration(gt_df)
    abs_error_seconds = float(abs(pred_seconds - gt_seconds))
    signed_error_seconds = float(pred_seconds - gt_seconds)
    rel_error = float(abs_error_seconds / gt_seconds) if gt_seconds > 0 else None
    overlap = overlap_metrics(
        pred_intervals_df,
        gt_df,
        float(summary["audio_duration_seconds"]),
        float(args.eval_resolution_seconds),
    )

    evaluation = {
        **summary,
        "annotations_csv": str(Path(args.annotations_csv).resolve()),
        "positive_labels": sorted(positive_labels),
        "ground_truth_piano_seconds": gt_seconds,
        "absolute_error_seconds": abs_error_seconds,
        "signed_error_seconds": signed_error_seconds,
        "relative_error_vs_ground_truth": rel_error,
        "ground_truth_num_intervals": int(len(gt_df)),
        "eval_resolution_seconds": float(args.eval_resolution_seconds),
        "overlap_metrics": overlap,
    }

    save_timeline_outputs(
        frame_df,
        pred_intervals_df,
        evaluation,
        out_frames_csv=args.out_frames_csv,
        out_intervals_csv=args.out_intervals_csv,
        out_summary_json=args.out_summary_json,
    )
    out_gt = Path(args.out_ground_truth_csv)
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(out_gt, index=False)

    print(f"audio_duration_seconds={summary['audio_duration_seconds']:.2f}")
    print(f"predicted_piano_seconds={pred_seconds:.2f}")
    print(f"ground_truth_piano_seconds={gt_seconds:.2f}")
    print(f"absolute_error_seconds={abs_error_seconds:.2f}")
    print(f"precision={overlap['precision']:.4f}")
    print(f"recall={overlap['recall']:.4f}")
    print(f"f1={overlap['f1']:.4f}")
    print(f"iou={overlap['iou']:.4f}")
    print(f"frames_csv={args.out_frames_csv}")
    print(f"intervals_csv={args.out_intervals_csv}")
    print(f"ground_truth_csv={args.out_ground_truth_csv}")
    print(f"summary_json={args.out_summary_json}")


if __name__ == "__main__":
    main()
