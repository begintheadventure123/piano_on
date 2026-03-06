import librosa
import numpy as np


def load_audio_mono(path: str, cfg: dict) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=True, duration=cfg["max_duration_seconds"])
    return np.asarray(y, dtype=np.float32), int(sr)


def _safe_stats(arr: np.ndarray) -> list[float]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return [0.0, 0.0]
    return [float(np.mean(arr)), float(np.std(arr))]


def _band_stats(arr: np.ndarray) -> list[float]:
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


def featurize_v2_waveform(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(175, dtype=np.float32)

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
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg["hop_length"])
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=cfg["hop_length"])[0])

    harmonic, percussive = librosa.effects.hpss(y)
    harm_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + eps))
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    features: list[float] = []
    for band in logmel:
        features.extend(_safe_stats(band))
    for block in (mfcc, mfcc_delta, mfcc_delta2, chroma, contrast, tonnetz):
        for band in block:
            features.extend(_safe_stats(band))
    for arr in (centroid, bandwidth, rolloff, flatness, zcr, rms):
        features.extend(_band_stats(arr))
    features.extend(_band_stats(onset_env))
    features.extend(
        [
            tempo,
            harm_ratio,
            float(np.max(np.abs(y))),
            float(np.mean(np.abs(y))),
            float(np.std(y)),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def featurize_v2(path: str, cfg: dict) -> np.ndarray:
    y, sr = load_audio_mono(path, cfg)
    return featurize_v2_waveform(y, sr, cfg)
