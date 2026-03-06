import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".wma", ".flac", ".aiff", ".aif"}


def rms_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(y), dtype=np.float64))
    return 20.0 * np.log10(float(rms) + 1e-12)


def iter_inputs(path: Path):
    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTS:
            yield path
        return
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Split long audio into fixed-length training clips")
    parser.add_argument("--input", required=True, help="Input audio file or folder")
    parser.add_argument("--out", required=True, help="Output folder for clips")
    parser.add_argument("--label-prefix", default="non_piano", help="Filename prefix for output clips")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate")
    parser.add_argument("--clip-seconds", type=float, default=8.0, help="Clip duration in seconds")
    parser.add_argument("--hop-seconds", type=float, default=8.0, help="Sliding hop duration in seconds")
    parser.add_argument(
        "--min-rms-db",
        type=float,
        default=-50.0,
        help="Drop segments below this RMS dB level (silence filter)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_len = int(args.sample_rate * args.clip_seconds)
    hop_len = int(args.sample_rate * args.hop_seconds)
    if clip_len <= 0 or hop_len <= 0:
        raise ValueError("clip-seconds and hop-seconds must be > 0")

    total_saved = 0
    total_seen = 0

    for audio_file in iter_inputs(in_path):
        total_seen += 1
        y, _ = librosa.load(str(audio_file), sr=args.sample_rate, mono=True)
        if y.size < clip_len:
            continue

        stem = audio_file.stem.replace(" ", "_")
        local_idx = 0
        for start in range(0, len(y) - clip_len + 1, hop_len):
            seg = y[start : start + clip_len]
            if rms_db(seg) < args.min_rms_db:
                continue
            name = f"{args.label_prefix}_{stem}_{local_idx:05d}.wav"
            sf.write(out_dir / name, seg, args.sample_rate, subtype="PCM_16")
            local_idx += 1
            total_saved += 1

    print(f"Processed inputs: {total_seen}")
    print(f"Saved clips: {total_saved}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
