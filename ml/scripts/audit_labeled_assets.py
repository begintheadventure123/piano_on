import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "ml" / "data" / "review" / "manual_label_archive"
CONSOLIDATED_CSV = ARCHIVE_ROOT / "consolidated_labels.csv"
OPS_DB_PATH = ROOT / "ml" / "data" / "ops" / "activity.db"
OUT_DIR = ROOT / "ml" / "data" / "review" / "organized_labels"
SUMMARY_JSON = OUT_DIR / "summary.json"
DETAIL_CSV = OUT_DIR / "labeled_asset_inventory.csv"
REUSABLE_CSV = OUT_DIR / "reusable_labels.csv"
MISSING_CSV = OUT_DIR / "missing_audio_labels.csv"
TRAIN_REUSABLE_CSV = OUT_DIR / "train_reusable_labels.csv"


def normalize_label(value: str) -> str:
    label = str(value or "").strip().lower()
    return label


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def build_review_copy_index() -> dict[str, str]:
    review_root = ROOT / "ml" / "data" / "review"
    if not review_root.exists():
        return {}
    out = {}
    for path in review_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        out.setdefault(name, str(path.resolve()))
        parts = name.split("_", 2)
        if len(parts) == 3:
            out.setdefault(parts[2], str(path.resolve()))
    return out


def load_promoted_index() -> dict[tuple[str, str], str]:
    if not OPS_DB_PATH.exists():
        return {}
    with sqlite3.connect(str(OPS_DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT source_path, label, target_path FROM promoted_files"
        ).fetchall()
    out = {}
    for source_path, label, target_path in rows:
        out[(str(source_path), str(label))] = str(target_path)
    return out


def load_holdout_sources() -> set[str]:
    holdout_csv = ROOT / "ml" / "data" / "eval" / "holdout" / "holdout_index.csv"
    if not holdout_csv.exists():
        return set()
    df = pd.read_csv(holdout_csv)
    out = set()
    for col in ("resolved_source_path", "source_path"):
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            if not value.strip():
                continue
            resolved = str(resolve_repo_path(value))
            out.add(resolved)
            out.add(str(value).strip())
    return out


def batch_has_tracking_artifact(row: pd.Series) -> bool:
    if str(row.get("dataset_type", "")) == "report_review":
        return bool(str(row.get("source_file", "")).strip())
    source_file = str(row.get("source_file", "")).strip()
    if not source_file:
        return False
    p = Path(source_file)
    batch_dir = p.parent
    for name in ("false_positive_review.csv", "false_positive_review_labeled.csv", "false_positive_review_labels.json"):
        if (batch_dir / name).exists():
            return True
    return False


def main() -> None:
    if not CONSOLIDATED_CSV.exists():
        raise FileNotFoundError(f"Missing consolidated labels: {CONSOLIDATED_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONSOLIDATED_CSV)
    df["human_label"] = df["human_label"].map(normalize_label)
    df = df[df["human_label"].isin(["piano", "non_piano", "unclear"])].copy()

    promoted_index = load_promoted_index()
    review_copy_index = build_review_copy_index()
    holdout_sources = load_holdout_sources()
    rows = []

    for row in df.itertuples(index=False):
        source_path_raw = str(getattr(row, "source_path", "") or "").strip()
        source_path_resolved = str(resolve_repo_path(source_path_raw)) if source_path_raw else ""
        label = str(getattr(row, "human_label", "") or "").strip()
        source_file = str(getattr(row, "source_file", "") or "").strip()
        dataset_id = str(getattr(row, "dataset_id", "") or "").strip()
        dataset_type = str(getattr(row, "dataset_type", "") or "").strip()
        source_exists = bool(source_path_resolved) and Path(source_path_resolved).exists()
        promoted_path = promoted_index.get((source_path_resolved, label), "")
        if not promoted_path and source_path_raw != source_path_resolved:
            promoted_path = promoted_index.get((source_path_raw, label), "")
        promoted_exists = bool(promoted_path) and Path(promoted_path).exists()
        review_copy_path = review_copy_index.get(Path(source_path_raw).name if source_path_raw else "", "")
        review_copy_exists = bool(review_copy_path) and Path(review_copy_path).exists()
        in_holdout = source_path_raw in holdout_sources or source_path_resolved in holdout_sources
        tracking_exists = batch_has_tracking_artifact(pd.Series({
            "dataset_type": dataset_type,
            "source_file": source_file,
        }))

        if label == "unclear":
            reuse_status = "not_reusable_unclear"
            reusable_path = ""
        elif promoted_exists:
            reuse_status = "reusable_promoted_copy"
            reusable_path = promoted_path
        elif review_copy_exists:
            reuse_status = "reusable_review_copy"
            reusable_path = review_copy_path
        elif source_exists:
            reuse_status = "reusable_source_exists"
            reusable_path = source_path_resolved
        else:
            reuse_status = "missing_audio"
            reusable_path = ""

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_type": dataset_type,
                "source_path": source_path_raw,
                "resolved_source_path": source_path_resolved,
                "source_exists": source_exists,
                "human_label": label,
                "source_file": source_file,
                "tracking_artifact_exists": tracking_exists,
                "promoted_path": promoted_path,
                "promoted_exists": promoted_exists,
                "review_copy_path": review_copy_path,
                "review_copy_exists": review_copy_exists,
                "in_fixed_holdout": in_holdout,
                "reuse_status": reuse_status,
                "reusable_audio_path": reusable_path,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(DETAIL_CSV, index=False)
    out_df[out_df["reuse_status"].isin(["reusable_promoted_copy", "reusable_review_copy", "reusable_source_exists"])].to_csv(REUSABLE_CSV, index=False)
    out_df[out_df["reuse_status"] == "missing_audio"].to_csv(MISSING_CSV, index=False)
    out_df[
        out_df["reuse_status"].isin(["reusable_promoted_copy", "reusable_review_copy", "reusable_source_exists"])
        & (~out_df["in_fixed_holdout"])
        & (out_df["human_label"] != "unclear")
    ].to_csv(TRAIN_REUSABLE_CSV, index=False)

    summary = {
        "consolidated_labels_csv": str(CONSOLIDATED_CSV),
        "detail_csv": str(DETAIL_CSV),
        "reusable_csv": str(REUSABLE_CSV),
        "train_reusable_csv": str(TRAIN_REUSABLE_CSV),
        "missing_csv": str(MISSING_CSV),
        "total_manual_labels": int(len(out_df)),
        "label_counts": {k: int(v) for k, v in out_df["human_label"].value_counts().to_dict().items()},
        "reuse_status_counts": {k: int(v) for k, v in out_df["reuse_status"].value_counts().to_dict().items()},
        "tracking_artifact_counts": {
            "with_tracking_artifact": int(out_df["tracking_artifact_exists"].sum()),
            "without_tracking_artifact": int((~out_df["tracking_artifact_exists"]).sum()),
        },
        "reusable_for_training": int(out_df["reuse_status"].isin(["reusable_promoted_copy", "reusable_review_copy", "reusable_source_exists"]).sum()),
        "train_reusable_excluding_holdout": int(
            (
                out_df["reuse_status"].isin(["reusable_promoted_copy", "reusable_review_copy", "reusable_source_exists"])
                & (~out_df["in_fixed_holdout"])
                & (out_df["human_label"] != "unclear")
            ).sum()
        ),
        "missing_audio_not_reusable": int((out_df["reuse_status"] == "missing_audio").sum()),
        "unclear_not_for_training": int((out_df["reuse_status"] == "not_reusable_unclear").sum()),
        "in_fixed_holdout": int(out_df["in_fixed_holdout"].sum()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
