import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REVIEW_ROOT = ROOT / "ml" / "data" / "review"
ORGANIZED_ROOT = REVIEW_ROOT / "organized_labels"
ASSET_POOL_ROOT = REVIEW_ROOT / "asset_pool"
ASSET_INDEX_CSV = ASSET_POOL_ROOT / "asset_index.csv"
TRAIN_INDEX_CSV = ASSET_POOL_ROOT / "train_index.csv"
HOLDOUT_INDEX_CSV = ASSET_POOL_ROOT / "holdout_index.csv"
UNCLEAR_INDEX_CSV = ASSET_POOL_ROOT / "unclear_index.csv"
SKIPPED_INDEX_CSV = ASSET_POOL_ROOT / "skipped_index.csv"


def short_token(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def resolve_existing_path(raw_path: str) -> Path | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path if path.exists() else None


def choose_source(row: pd.Series) -> tuple[Path | None, str]:
    for col, kind in (
        ("promoted_path", "promoted"),
        ("review_copy_path", "review_copy"),
        ("resolved_source_path", "source"),
        ("source_path", "source"),
    ):
        path = resolve_existing_path(str(row.get(col, "")))
        if path:
            return path, kind
    return None, ""


def classify_split(row: pd.Series) -> str:
    label = str(row.get("human_label", "")).strip().lower()
    if label == "unclear":
        return "unclear"
    if bool(row.get("in_fixed_holdout", False)):
        return "holdout"
    return "train"


def safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "asset"


def materialize_asset_pool() -> dict:
    inventory_csv = ORGANIZED_ROOT / "labeled_asset_inventory.csv"
    if not inventory_csv.exists():
        raise FileNotFoundError(f"Missing labeled asset inventory: {inventory_csv}")

    df = pd.read_csv(inventory_csv)
    ASSET_POOL_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_rows = []
    copied = 0
    reused = 0
    skipped = 0

    for row in df.itertuples(index=False):
        series = pd.Series(row._asdict())
        label = str(series.get("human_label", "")).strip().lower()
        if label not in {"piano", "non_piano", "unclear"}:
            skipped += 1
            continue

        source_path, source_kind = choose_source(series)
        if source_path is None:
            skipped_rows.append(
                {
                    "dataset_id": str(series.get("dataset_id", "")),
                    "human_label": label,
                    "reuse_status": str(series.get("reuse_status", "")),
                    "in_fixed_holdout": bool(series.get("in_fixed_holdout", False)),
                    "original_source_path": str(series.get("source_path", "")),
                }
            )
            skipped += 1
            continue

        split = classify_split(series)
        dataset_id = safe_name(str(series.get("dataset_id", "")) or "unknown")
        source_name = safe_name(source_path.name)
        token = short_token(
            str(series.get("dataset_id", "")),
            str(series.get("source_path", "")),
            label,
            split,
        )
        target_dir = ASSET_POOL_ROOT / split / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{dataset_id}__{token}__{source_name}"

        if target_path.exists():
            reused += 1
        else:
            shutil.copy2(source_path, target_path)
            copied += 1

        rows.append(
            {
                "split": split,
                "human_label": label,
                "dataset_id": str(series.get("dataset_id", "")),
                "reuse_status": str(series.get("reuse_status", "")),
                "source_kind": source_kind,
                "in_fixed_holdout": bool(series.get("in_fixed_holdout", False)),
                "managed_path": str(target_path.resolve()),
                "canonical_source_path": str(source_path.resolve()),
                "original_source_path": str(series.get("source_path", "")),
                "promoted_path": str(series.get("promoted_path", "")),
                "review_copy_path": str(series.get("review_copy_path", "")),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        by=["split", "human_label", "dataset_id", "managed_path"],
        ascending=[True, True, True, True],
    )
    out_df.to_csv(ASSET_INDEX_CSV, index=False)
    out_df[out_df["split"] == "train"].to_csv(TRAIN_INDEX_CSV, index=False)
    out_df[out_df["split"] == "holdout"].to_csv(HOLDOUT_INDEX_CSV, index=False)
    out_df[out_df["split"] == "unclear"].to_csv(UNCLEAR_INDEX_CSV, index=False)
    pd.DataFrame(skipped_rows).to_csv(SKIPPED_INDEX_CSV, index=False)

    return {
        "asset_pool_root": str(ASSET_POOL_ROOT),
        "asset_index_csv": str(ASSET_INDEX_CSV),
        "train_index_csv": str(TRAIN_INDEX_CSV),
        "holdout_index_csv": str(HOLDOUT_INDEX_CSV),
        "unclear_index_csv": str(UNCLEAR_INDEX_CSV),
        "skipped_index_csv": str(SKIPPED_INDEX_CSV),
        "managed_assets": int(len(out_df)),
        "copied_assets": int(copied),
        "reused_assets": int(reused),
        "skipped_assets": int(skipped),
    }


def main() -> None:
    result = materialize_asset_pool()
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
