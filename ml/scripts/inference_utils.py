from pathlib import Path

import joblib
import numpy as np
import yaml

from model_utils import apply_platt, featurize, featurize_waveform
from model_utils_v2 import featurize_v2
from model_utils_v2 import featurize_v2_waveform
from model_utils_v3 import featurize_v3, featurize_v3_waveform


def load_model_pack(model_path: str | Path) -> dict:
    pack = joblib.load(model_path)
    if not isinstance(pack, dict):
        raise RuntimeError(f"Unsupported model pack format: {model_path}")
    return pack


def load_effective_config(pack: dict, config_path: str | Path | None = None) -> dict:
    file_cfg = None
    if config_path is not None:
        file_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pack_cfg = pack.get("config")
    if isinstance(pack_cfg, dict):
        merged = dict(file_cfg or {})
        merged.update(pack_cfg)
        return merged
    if file_cfg is None:
        raise RuntimeError("config path is required when model pack does not embed config")
    return file_cfg


def feature_mode_from_pack(pack: dict) -> str:
    return str(pack.get("feature_mode", "baseline_v1")).strip().lower()


def featurize_for_pack(audio_path: str, pack: dict, cfg: dict) -> np.ndarray:
    mode = feature_mode_from_pack(pack)
    if mode == "baseline_v1":
        return featurize(audio_path, cfg)
    if mode == "rich_v2":
        return featurize_v2(audio_path, cfg)
    if mode == "temporal_v3":
        return featurize_v3(audio_path, cfg)
    if mode == "ensemble_v1":
        return np.zeros(1, dtype=np.float32)
    raise RuntimeError(f"Unsupported feature mode: {mode}")


def featurize_waveform_for_pack(y: np.ndarray, sr: int, pack: dict, cfg: dict) -> np.ndarray:
    mode = feature_mode_from_pack(pack)
    if mode == "baseline_v1":
        return featurize_waveform(y, sr, cfg)
    if mode == "rich_v2":
        return featurize_v2_waveform(y, sr, cfg)
    if mode == "temporal_v3":
        return featurize_v3_waveform(y, sr, cfg)
    if mode == "ensemble_v1":
        return np.zeros(1, dtype=np.float32)
    raise RuntimeError(f"Unsupported feature mode: {mode}")


def calibrate_probabilities(raw_prob: np.ndarray, pack: dict) -> np.ndarray:
    calib = pack.get("calibration", {}) if isinstance(pack, dict) else {}
    enabled = bool(calib.get("enabled", False))
    if not enabled:
        return np.asarray(raw_prob, dtype=np.float64)
    method = str(calib.get("method", "none")).strip().lower()
    raw_prob = np.asarray(raw_prob, dtype=np.float64)
    if method == "platt":
        coef = float(calib.get("coef", 1.0))
        intercept = float(calib.get("intercept", 0.0))
        return apply_platt(raw_prob, coef, intercept)
    if method == "isotonic":
        calibrator = calib.get("model")
        if calibrator is None:
            raise RuntimeError("Missing isotonic calibrator model in pack")
        return np.asarray(calibrator.predict(raw_prob), dtype=np.float64)
    raise RuntimeError(f"Unsupported calibration method: {method}")


def _normalize_member(member: dict) -> dict:
    normalized = dict(member)
    if isinstance(normalized.get("pack"), dict):
        return normalized
    model_path = normalized.get("model_path")
    if not model_path:
        raise RuntimeError("Ensemble member is missing model_path")
    normalized["pack"] = load_model_pack(model_path)
    return normalized


def _aggregate_member_probs(member_probs: list[tuple[float, float]], pack: dict) -> tuple[float, float]:
    if not member_probs:
        raise RuntimeError("Ensemble has no member probabilities")
    method = str(pack.get("ensemble_method", "weighted_average")).strip().lower()
    raw_vals = np.asarray([p[0] for p in member_probs], dtype=np.float64)
    prob_vals = np.asarray([p[1] for p in member_probs], dtype=np.float64)
    weights = np.asarray(pack.get("ensemble_weights", [1.0] * len(member_probs)), dtype=np.float64)
    if weights.size != prob_vals.size or np.sum(weights) <= 0:
        weights = np.ones_like(prob_vals, dtype=np.float64)
    weights = weights / np.sum(weights)
    if method == "min":
        return float(np.min(raw_vals)), float(np.min(prob_vals))
    if method == "max":
        return float(np.max(raw_vals)), float(np.max(prob_vals))
    raw_prob = float(np.sum(raw_vals * weights))
    prob = float(np.sum(prob_vals * weights))
    return raw_prob, prob


def predict_probability(audio_path: str, pack: dict, cfg: dict | None) -> tuple[float, float]:
    if feature_mode_from_pack(pack) == "ensemble_v1":
        members = [_normalize_member(member) for member in pack.get("members", [])]
        member_probs = []
        for member in members:
            member_pack = member["pack"]
            member_cfg = load_effective_config(member_pack, member.get("config_path"))
            member_probs.append(predict_probability(audio_path, member_pack, member_cfg))
        return _aggregate_member_probs(member_probs, pack)
    model = pack["model"]
    x = featurize_for_pack(audio_path, pack, cfg).reshape(1, -1)
    raw_prob = float(model.predict_proba(x)[0, 1])
    prob = float(calibrate_probabilities(np.array([raw_prob]), pack)[0])
    return raw_prob, prob


def predict_probability_waveform(y: np.ndarray, sr: int, pack: dict, cfg: dict | None) -> tuple[float, float]:
    if feature_mode_from_pack(pack) == "ensemble_v1":
        members = [_normalize_member(member) for member in pack.get("members", [])]
        member_probs = []
        for member in members:
            member_pack = member["pack"]
            member_cfg = load_effective_config(member_pack, member.get("config_path"))
            member_probs.append(predict_probability_waveform(y, sr, member_pack, member_cfg))
        return _aggregate_member_probs(member_probs, pack)
    model = pack["model"]
    x = featurize_waveform_for_pack(y, sr, pack, cfg).reshape(1, -1)
    raw_prob = float(model.predict_proba(x)[0, 1])
    prob = float(calibrate_probabilities(np.array([raw_prob]), pack)[0])
    return raw_prob, prob


def decision_threshold_from_pack(pack: dict) -> float:
    return float(pack.get("decision_threshold", 0.5))


def review_threshold_from_pack(pack: dict) -> float:
    return float(pack.get("review_threshold", pack.get("decision_threshold", 0.5)))
