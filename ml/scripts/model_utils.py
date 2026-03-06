import math
import re
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def featurize(path: str, cfg: dict) -> np.ndarray:
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=True, duration=cfg["max_duration_seconds"])
    return featurize_waveform(y, sr, cfg)


def featurize_waveform(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(13, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=2.0,
    )
    logmel = librosa.power_to_db(mel + 1e-10)
    feats = [
        float(np.mean(logmel)),
        float(np.std(logmel)),
        float(np.percentile(logmel, 10)),
        float(np.percentile(logmel, 50)),
        float(np.percentile(logmel, 90)),
    ]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    for arr in (centroid, rolloff, zcr, rms):
        feats.append(float(np.mean(arr)))
        feats.append(float(np.std(arr)))
    # A simple robustness feature for near-silence or heavily compressed clips.
    feats.append(float(np.max(np.abs(y))))
    return np.array(feats, dtype=np.float32)


def source_group_key(path: str) -> str:
    p = Path(path)
    stem = p.stem
    stem = re.sub(r"_[0-9]{3,6}$", "", stem)
    return f"{str(p.parent)}|{stem.lower()}"


def grouped_split_indices(
    y: np.ndarray,
    groups: list[str],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train/val/test sizes must sum to 1.0")

    groups_arr = np.array(groups)
    idx = np.arange(len(y))

    global_pos = float(np.mean(y))
    best = None
    best_score = float("inf")

    rel_val = val_size / (val_size + test_size)
    for i in range(100):
        first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed + i)
        train_idx, temp_idx = next(first.split(idx, y, groups=groups_arr))

        temp_groups = groups_arr[temp_idx]
        second = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1000 + i)
        val_rel, test_rel = next(second.split(temp_idx, y[temp_idx], groups=temp_groups))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        splits = [train_idx, val_idx, test_idx]
        bad = False
        score = 0.0
        for s in splits:
            ys = y[s]
            if len(ys) == 0 or len(np.unique(ys)) < 2:
                bad = True
                break
            pos = float(np.mean(ys))
            score += abs(pos - global_pos)
        if bad:
            continue
        # Keep test/val reasonably sized.
        score += abs(len(val_idx) - (len(y) * val_size)) / max(len(y), 1)
        score += abs(len(test_idx) - (len(y) * test_size)) / max(len(y), 1)
        if score < best_score:
            best_score = score
            best = (train_idx, val_idx, test_idx)

    if best is None:
        raise RuntimeError("Unable to produce a valid grouped split after retries")
    return best


def stratified_fallback_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=y,
    )
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio),
        random_state=seed,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_platt(prob: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    logits = coef * p + intercept
    return sigmoid(logits)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y_true), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += abs(acc - conf) * (float(np.sum(mask)) / n)
    return float(ece)


def fpr_at_recall(y_true: np.ndarray, prob: np.ndarray, recall_target: float) -> float | None:
    fpr, tpr, _ = roc_curve(y_true, prob)
    valid = np.where(tpr >= recall_target)[0]
    if valid.size == 0:
        return None
    return float(np.min(fpr[valid]))


def choose_threshold_for_recall(y_true: np.ndarray, prob: np.ndarray, recall_target: float, default: float = 0.5) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    valid = np.where(tpr >= recall_target)[0]
    if valid.size == 0:
        return float(default)
    best = valid[np.argmin(fpr[valid])]
    thr = float(thresholds[best])
    if not math.isfinite(thr):
        return float(default)
    return float(min(max(thr, 0.0), 1.0))


def choose_threshold_balanced_accuracy(
    y_true: np.ndarray,
    prob: np.ndarray,
    default: float = 0.5,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if y_true.size == 0 or prob.size == 0:
        return float(default)

    candidates = np.unique(np.clip(prob, min_threshold, max_threshold))
    candidates = np.concatenate(([min_threshold], candidates, [max_threshold]))
    best_thr = float(default)
    best_score = -1.0
    for thr in candidates:
        pred = (prob >= float(thr)).astype(np.int64)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        score = 0.5 * (tpr + tnr)
        if score > best_score or (math.isclose(score, best_score) and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score = score
            best_thr = float(thr)
    return float(min(max(best_thr, min_threshold), max_threshold))


def build_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float, recall_target: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    out = {
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_true, prob)),
        "auc_pr": float(average_precision_score(y_true, prob)),
        "ece_10": expected_calibration_error(y_true, prob, n_bins=10),
        "classification_report": classification_report(y_true, pred, output_dict=True),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "fpr_at_recall_target": fpr_at_recall(y_true, prob, recall_target),
        "recall_target": float(recall_target),
    }
    return out
