import argparse
import csv
import shutil
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_CSV = ROOT / "ml" / "data" / "review" / "manual_label_archive" / "consolidated_labels.csv"
OPS_DB = ROOT / "ml" / "data" / "ops" / "activity.db"


def load_promoted_lookup(db_path: Path) -> dict[tuple[str, str], str]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT source_path, label, target_path FROM promoted_files").fetchall()
    return {(str(src), str(label)): str(target) for src, label, target in rows}


def resolve_audio_path(source_path: str, human_label: str, promoted_lookup: dict[tuple[str, str], str]) -> str:
    src = Path(str(source_path))
    if src.exists():
        return str(src.resolve())
    promoted = promoted_lookup.get((str(source_path), str(human_label)))
    if promoted and Path(promoted).exists():
        return str(Path(promoted).resolve())
    return ""


def stable_name(dataset_id: str, rank: int, score: float, source_path: str) -> str:
    src = Path(source_path)
    suffix = src.suffix or ".wav"
    safe_dataset = dataset_id.replace(" ", "_")
    return f"{rank:03d}_{safe_dataset}_s{score:.4f}_{src.stem}{suffix}"


def copy_selection(rows: pd.DataFrame, dst_dir: Path, label_name: str) -> list[dict]:
    copied = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows.itertuples(index=False), start=1):
        resolved = Path(str(row.resolved_source_path))
        if not resolved.exists():
            continue
        score = float(row.selection_score)
        out_name = stable_name(str(row.dataset_id), rank, score, str(row.source_path))
        out_path = dst_dir / out_name
        shutil.copy2(resolved, out_path)
        copied.append(
            {
                "label": label_name,
                "dataset_id": str(row.dataset_id),
                "source_path": str(row.source_path),
                "resolved_source_path": str(resolved),
                "holdout_path": str(out_path),
                "human_label": str(row.human_label),
                "piano_probability": float(row.piano_probability),
                "selection_score": score,
                "predicted_label": str(row.predicted_label),
            }
        )
    return copied


def main():
    parser = argparse.ArgumentParser(description="Build fixed holdout set from manually labeled hard examples")
    parser.add_argument("--archive-csv", default=str(ARCHIVE_CSV))
    parser.add_argument("--out-root", default="ml/data/eval/holdout")
    parser.add_argument("--exclude-out", default="ml/data/eval/holdout_exclude_paths.txt")
    parser.add_argument("--top-piano", type=int, default=40, help="number of hardest true-piano samples to hold out")
    parser.add_argument("--top-non-piano", type=int, default=40, help="number of hardest true-non-piano samples to hold out")
    args = parser.parse_args()

    archive_csv = Path(args.archive_csv)
    if not archive_csv.exists():
        raise FileNotFoundError(f"Archive CSV not found: {archive_csv}")

    df = pd.read_csv(archive_csv)
    if df.empty:
        raise RuntimeError("Archive CSV is empty")

    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    df["piano_probability"] = pd.to_numeric(df["piano_probability"], errors="coerce")
    df = df.dropna(subset=["piano_probability"]).copy()
    promoted_lookup = load_promoted_lookup(OPS_DB)
    df["resolved_source_path"] = df.apply(
        lambda r: resolve_audio_path(r["source_path"], r["human_label"], promoted_lookup),
        axis=1,
    )
    df = df[df["resolved_source_path"].astype(str).str.len() > 0].copy()
    df["resolved_source_path"] = df["resolved_source_path"].map(lambda p: str(Path(p).resolve()))
    df = df.drop_duplicates(subset=["resolved_source_path", "human_label"], keep="first").copy()

    piano_df = df[df["human_label"] == "piano"].copy()
    piano_df["selection_score"] = -piano_df["piano_probability"]
    piano_df = piano_df.sort_values(by=["piano_probability", "dataset_id", "source_path"], ascending=[True, True, True])

    non_piano_df = df[df["human_label"] == "non_piano"].copy()
    non_piano_df["selection_score"] = non_piano_df["piano_probability"]
    non_piano_df = non_piano_df.sort_values(by=["piano_probability", "dataset_id", "source_path"], ascending=[False, True, True])

    piano_pick = piano_df.head(max(0, int(args.top_piano))).copy()
    non_piano_pick = non_piano_df.head(max(0, int(args.top_non_piano))).copy()

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    copied_rows = []
    copied_rows.extend(copy_selection(piano_pick, out_root / "piano", "piano"))
    copied_rows.extend(copy_selection(non_piano_pick, out_root / "non_piano", "non_piano"))

    if not copied_rows:
        raise RuntimeError("No holdout files could be copied from manually labeled assets")

    index_csv = out_root / "holdout_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "dataset_id",
                "source_path",
                "resolved_source_path",
                "holdout_path",
                "human_label",
                "piano_probability",
                "selection_score",
                "predicted_label",
            ],
        )
        writer.writeheader()
        for row in copied_rows:
            writer.writerow(row)

    exclude_out = Path(args.exclude_out)
    exclude_out.parent.mkdir(parents=True, exist_ok=True)
    exclude_out.write_text(
        "\n".join(sorted({row["resolved_source_path"] for row in copied_rows})) + "\n",
        encoding="utf-8",
    )

    print(f"holdout_root={out_root.resolve()}")
    print(f"holdout_rows={len(copied_rows)}")
    print(f"piano_rows={sum(1 for row in copied_rows if row['label'] == 'piano')}")
    print(f"non_piano_rows={sum(1 for row in copied_rows if row['label'] == 'non_piano')}")
    print(f"exclude_paths={exclude_out.resolve()}")


if __name__ == "__main__":
    main()
