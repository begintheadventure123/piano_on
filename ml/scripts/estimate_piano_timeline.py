import argparse
import json
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from inference_utils import (
    decision_threshold_from_pack,
    load_effective_config,
    load_model_pack,
    predict_probability_waveform,
)


warnings.filterwarnings(
    "ignore",
    message=r"n_fft=.*is too large for input signal of length=.*",
    category=UserWarning,
)


def resolve_timeline_thresholds(enter_threshold: float | None, exit_threshold: float | None, decision_threshold: float) -> tuple[float, float]:
    default_enter_threshold = min(float(decision_threshold), 0.70)
    enter = float(enter_threshold if enter_threshold is not None else default_enter_threshold)
    exit_ = float(exit_threshold if exit_threshold is not None else max(0.0, enter - 0.10))
    if exit_ > enter:
        raise ValueError("exit-threshold must be <= enter-threshold")
    return enter, exit_


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or window_size <= 1:
        return values.copy()
    window_size = max(1, int(window_size))
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")


def build_windows(y: np.ndarray, sr: int, window_seconds: float, hop_seconds: float) -> list[dict]:
    window_samples = max(1, int(round(window_seconds * sr)))
    hop_samples = max(1, int(round(hop_seconds * sr)))
    if y.size == 0:
        return []

    rows: list[dict] = []
    start = 0
    while start < y.size:
        end = min(start + window_samples, y.size)
        seg = y[start:end]
        if seg.size < window_samples:
            seg = np.pad(seg, (0, window_samples - seg.size))
        rows.append(
            {
                "start_sample": int(start),
                "end_sample": int(end),
                "start_seconds": float(start / sr),
                "end_seconds": float(end / sr),
                "segment": seg.astype(np.float32, copy=False),
            }
        )
        if end >= y.size:
            break
        start += hop_samples
    return rows


def detect_intervals(
    frame_df: pd.DataFrame,
    *,
    enter_threshold: float,
    exit_threshold: float,
    min_interval_seconds: float,
    merge_gap_seconds: float,
    audio_duration_seconds: float,
) -> list[dict]:
    intervals: list[dict] = []
    active = False
    active_start = 0.0
    active_end = 0.0

    for row in frame_df.itertuples(index=False):
        score = float(row.piano_probability_smoothed)
        start_seconds = float(row.start_seconds)
        end_seconds = float(row.end_seconds)
        if not active:
            if score >= enter_threshold:
                active = True
                active_start = start_seconds
                active_end = end_seconds
        else:
            active_end = end_seconds
            if score < exit_threshold:
                intervals.append({"start_seconds": active_start, "end_seconds": active_end})
                active = False

    if active:
        intervals.append({"start_seconds": active_start, "end_seconds": min(active_end, audio_duration_seconds)})

    merged: list[dict] = []
    for interval in intervals:
        start_seconds = float(interval["start_seconds"])
        end_seconds = float(interval["end_seconds"])
        if end_seconds - start_seconds < min_interval_seconds:
            continue
        if merged and start_seconds - float(merged[-1]["end_seconds"]) <= merge_gap_seconds:
            merged[-1]["end_seconds"] = max(float(merged[-1]["end_seconds"]), end_seconds)
            continue
        merged.append({"start_seconds": start_seconds, "end_seconds": end_seconds})

    for idx, interval in enumerate(merged, start=1):
        interval["duration_seconds"] = float(interval["end_seconds"] - interval["start_seconds"])
        interval["interval_index"] = idx
    return merged


def interval_for_time(intervals: list[dict], start_seconds: float, end_seconds: float) -> int | None:
    for interval in intervals:
        if float(interval["end_seconds"]) <= start_seconds:
            continue
        if float(interval["start_seconds"]) >= end_seconds:
            break
        return int(interval["interval_index"])
    return None


def load_audio_for_timeline(audio_path: str | Path, sample_rate: int) -> tuple[Path, np.ndarray, int, float]:
    resolved = Path(audio_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file does not exist: {resolved}")
    y, sr = librosa.load(str(resolved), sr=int(sample_rate), mono=True)
    y = np.asarray(y, dtype=np.float32)
    duration_seconds = float(y.size / max(sr, 1))
    return resolved, y, sr, duration_seconds


def estimate_piano_timeline(
    audio_path: str | Path,
    *,
    model_path: str | Path = "ml/models/piano_detector_v23_ensemble.joblib",
    config_path: str | Path = "ml/configs/train_config_v2.yaml",
    window_seconds: float = 2.0,
    hop_seconds: float = 0.5,
    smooth_frames: int = 5,
    enter_threshold: float | None = None,
    exit_threshold: float | None = None,
    min_interval_seconds: float = 1.0,
    merge_gap_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window-seconds and hop-seconds must be > 0")

    pack = load_model_pack(model_path)
    cfg = load_effective_config(pack, config_path)
    sample_rate = int(cfg["sample_rate"])
    decision_threshold = decision_threshold_from_pack(pack)
    enter, exit_ = resolve_timeline_thresholds(enter_threshold, exit_threshold, decision_threshold)

    resolved_audio_path, y, sr, audio_duration_seconds = load_audio_for_timeline(audio_path, sample_rate)
    windows = build_windows(y, sr, float(window_seconds), float(hop_seconds))
    if not windows:
        raise RuntimeError(f"No audio content could be loaded from {resolved_audio_path}")

    frame_rows: list[dict] = []
    for row in windows:
        raw_prob, prob = predict_probability_waveform(row["segment"], sr, pack, cfg)
        frame_rows.append(
            {
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "window_seconds": float(row["end_seconds"] - row["start_seconds"]),
                "piano_probability_raw": float(raw_prob),
                "piano_probability": float(prob),
            }
        )

    frame_df = pd.DataFrame(frame_rows)
    frame_df["piano_probability_smoothed"] = moving_average(
        frame_df["piano_probability"].to_numpy(dtype=np.float64),
        int(smooth_frames),
    )

    intervals = detect_intervals(
        frame_df,
        enter_threshold=enter,
        exit_threshold=exit_,
        min_interval_seconds=float(min_interval_seconds),
        merge_gap_seconds=float(merge_gap_seconds),
        audio_duration_seconds=audio_duration_seconds,
    )

    frame_df["interval_index"] = [
        interval_for_time(intervals, float(start), float(end))
        for start, end in zip(frame_df["start_seconds"], frame_df["end_seconds"])
    ]
    frame_df["is_piano_active"] = frame_df["interval_index"].notna().astype(int)

    intervals_df = pd.DataFrame(intervals)
    if intervals_df.empty:
        intervals_df = pd.DataFrame(columns=["interval_index", "start_seconds", "end_seconds", "duration_seconds"])
    else:
        intervals_df = intervals_df[["interval_index", "start_seconds", "end_seconds", "duration_seconds"]]

    total_piano_seconds = float(intervals_df["duration_seconds"].sum()) if not intervals_df.empty else 0.0
    coverage_ratio = float(total_piano_seconds / audio_duration_seconds) if audio_duration_seconds > 0 else 0.0
    summary = {
        "audio_path": str(resolved_audio_path.resolve()),
        "model": str(Path(model_path).resolve()),
        "audio_duration_seconds": audio_duration_seconds,
        "model_decision_threshold": float(decision_threshold),
        "estimated_piano_seconds": total_piano_seconds,
        "estimated_non_piano_seconds": float(max(0.0, audio_duration_seconds - total_piano_seconds)),
        "estimated_piano_ratio": coverage_ratio,
        "num_intervals": int(len(intervals)),
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "smooth_frames": int(smooth_frames),
        "enter_threshold": enter,
        "exit_threshold": exit_,
        "min_interval_seconds": float(min_interval_seconds),
        "merge_gap_seconds": float(merge_gap_seconds),
        "intervals": intervals_df.to_dict(orient="records"),
    }
    return frame_df, intervals_df, summary


def save_timeline_outputs(frame_df: pd.DataFrame, intervals_df: pd.DataFrame, summary: dict, *, out_frames_csv: str | Path, out_intervals_csv: str | Path, out_summary_json: str | Path) -> None:
    out_frames = Path(out_frames_csv)
    out_intervals = Path(out_intervals_csv)
    out_summary = Path(out_summary_json)
    out_frames.parent.mkdir(parents=True, exist_ok=True)
    out_intervals.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(out_frames, index=False)
    intervals_df.to_csv(out_intervals, index=False)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate piano timeline and total piano duration in a long audio file")
    parser.add_argument("audio_path", help="Long audio file to scan")
    parser.add_argument("--model", default="ml/models/piano_detector_v23_ensemble.joblib")
    parser.add_argument("--config", default="ml/configs/train_config_v2.yaml")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=5,
        help="Centered moving-average window over frame probabilities",
    )
    parser.add_argument(
        "--enter-threshold",
        type=float,
        default=None,
        help="Probability threshold to enter piano-active state; default is tuned lower than clip threshold for timeline scanning",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help="Probability threshold to exit piano-active state; default is enter-threshold - 0.10",
    )
    parser.add_argument("--min-interval-seconds", type=float, default=1.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--out-frames-csv", default="ml/reports/piano_timeline_frames.csv")
    parser.add_argument("--out-intervals-csv", default="ml/reports/piano_timeline_intervals.csv")
    parser.add_argument("--out-summary-json", default="ml/reports/piano_timeline_summary.json")
    args = parser.parse_args()

    frame_df, intervals_df, summary = estimate_piano_timeline(
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
    save_timeline_outputs(
        frame_df,
        intervals_df,
        summary,
        out_frames_csv=args.out_frames_csv,
        out_intervals_csv=args.out_intervals_csv,
        out_summary_json=args.out_summary_json,
    )

    print(f"audio_duration_seconds={summary['audio_duration_seconds']:.2f}")
    print(f"estimated_piano_seconds={summary['estimated_piano_seconds']:.2f}")
    print(f"estimated_piano_ratio={summary['estimated_piano_ratio']:.4f}")
    print(f"num_intervals={summary['num_intervals']}")
    print(f"enter_threshold={summary['enter_threshold']:.4f}")
    print(f"exit_threshold={summary['exit_threshold']:.4f}")
    print(f"frames_csv={args.out_frames_csv}")
    print(f"intervals_csv={args.out_intervals_csv}")
    print(f"summary_json={args.out_summary_json}")


if __name__ == "__main__":
    main()
