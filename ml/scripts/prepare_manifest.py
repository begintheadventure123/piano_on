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


def load_excluded_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(Path(line.strip()).resolve())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main():
    parser = argparse.ArgumentParser(description="Build training manifest from data folders")
    parser.add_argument("--root", default="ml/data/raw", help="raw data root")
    parser.add_argument("--out", default="ml/data/labels/manifest.csv", help="output manifest")
    parser.add_argument(
        "--exclude-paths",
        default="ml/data/eval/holdout_exclude_paths.txt",
        help="optional newline-delimited source paths to exclude from training",
    )
    parser.add_argument(
        "--extra-exclude-paths",
        default="ml/data/review/asset_pool/relabel_exclude_paths.txt",
        help="optional second newline-delimited source paths to exclude from training",
    )
    args = parser.parse_args()

    root = Path(args.root)
    excluded_paths = load_excluded_paths(Path(args.exclude_paths))
    excluded_paths.update(load_excluded_paths(Path(args.extra_exclude_paths)))
    rows = []
    rows.extend(collect(root / "piano", 1))
    rows.extend(collect(root / "non_piano", 0))
    rows.extend(collect(root / "mixed", 1))

    if not rows:
        raise RuntimeError("No audio files found under ml/data/raw")

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"])
    before_exclude = int(len(df))
    if excluded_paths:
        df = df[df["path"].map(lambda p: str(Path(p).resolve()) not in excluded_paths)].copy()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"manifest: {out}")
    print(f"excluded_holdout_sources={before_exclude - len(df)}")
    print(df.label.value_counts().sort_index().rename(index={0: 'non_piano', 1: 'piano_or_mixed'}))


if __name__ == "__main__":
    main()
