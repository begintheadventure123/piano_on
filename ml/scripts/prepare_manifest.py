import argparse
from pathlib import Path
import pandas as pd

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".wma", ".flac", ".aiff", ".aif"}


def collect(split_dir: Path, label: int):
    rows = []
    if not split_dir.exists():
        return rows
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            rows.append({"path": str(p), "label": label})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build training manifest from data folders")
    parser.add_argument("--root", default="ml/data/raw", help="raw data root")
    parser.add_argument("--out", default="ml/data/labels/manifest.csv", help="output manifest")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    rows.extend(collect(root / "piano", 1))
    rows.extend(collect(root / "non_piano", 0))
    rows.extend(collect(root / "mixed", 1))

    if not rows:
        raise RuntimeError("No audio files found under ml/data/raw")

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"manifest: {out}")
    print(df.label.value_counts().sort_index().rename(index={0: 'non_piano', 1: 'piano_or_mixed'}))


if __name__ == "__main__":
    main()
