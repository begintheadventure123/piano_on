import librosa
import numpy as np

from model_utils_v2 import load_audio_mono


def _temporal_stats(arr: np.ndarray) -> list[float]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return [0.0] * 6
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
        float(np.max(arr)),
    ]


def _burst_stats(mat: np.ndarray) -> list[float]:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    peak = np.max(mat, axis=1)
    mean = np.mean(mat, axis=1)
    p90 = np.percentile(mat, 90, axis=1)
    return [
        float(np.mean(peak - mean)),
        float(np.std(peak - mean)),
        float(np.mean(p90 - mean)),
        float(np.std(p90 - mean)),
    ]


def featurize_v3(path: str, cfg: dict) -> np.ndarray:
    y, sr = load_audio_mono(path, cfg)
    return featurize_v3_waveform(y, sr, cfg)


def featurize_v3_waveform(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(764, dtype=np.float32)

    eps = 1e-10
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=2.0,
    )
    logmel = librosa.power_to_db(mel + eps)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=int(cfg.get("n_mfcc", 20)))
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg["hop_length"])
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=cfg["hop_length"])[0])

    features: list[float] = []
    for block in (logmel, mfcc, mfcc_delta, chroma):
        for band in block:
            features.extend(_temporal_stats(band))
        features.extend(_burst_stats(block))

    for arr in (centroid, bandwidth, rolloff, flatness, zcr, rms):
        features.extend(_temporal_stats(arr))
    features.extend(_temporal_stats(onset_env))

    onset_thresh = float(np.percentile(onset_env, 90)) if onset_env.size else 0.0
    onset_peaks = int(np.sum(onset_env >= onset_thresh)) if onset_env.size else 0
    onset_density = float(onset_peaks / max(1, onset_env.size))
    duration_seconds = float(y.size / max(1, sr))
    features.extend(
        [
            tempo,
            duration_seconds,
            float(np.max(np.abs(y))),
            float(np.mean(np.abs(y))),
            float(np.std(y)),
            float(onset_peaks),
            onset_density,
            float(np.mean(logmel.max(axis=0))) if logmel.size else 0.0,
            float(np.std(logmel.max(axis=0))) if logmel.size else 0.0,
            float(np.mean(rms >= np.percentile(rms, 90))) if rms.size else 0.0,
            float(np.mean(chroma.max(axis=0))) if chroma.size else 0.0,
        ]
    )
    return np.asarray(features, dtype=np.float32)
