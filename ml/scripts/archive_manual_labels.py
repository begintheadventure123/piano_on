import csv
import json
import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INBOX_ROOT = ROOT / "ml" / "data" / "inbox"
REPORTS_ROOT = ROOT / "ml" / "reports"
OPS_DB = ROOT / "ml" / "data" / "ops" / "activity.db"
ARCHIVE_ROOT = ROOT / "ml" / "data" / "review" / "manual_label_archive"


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_label(value: str) -> str:
    return str(value or "").strip().lower()


def copy_if_exists(src: Path, dst_root: Path) -> str:
    if not src.exists() or not src.is_file():
        return ""
    dst = dst_root / src.relative_to(ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def collect_inbox_batch_rows(batch_dir: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    batch_name = batch_dir.name
    review_csv = batch_dir / "false_positive_review.csv"
    labeled_csv = batch_dir / "false_positive_review_labeled.csv"
    labels_json = batch_dir / "false_positive_review_labels.json"
    finalize_summary = batch_dir / "finalize_summary.json"
    iteration_summary = batch_dir / "iteration_summary.json"

    metadata = {
        "dataset_id": batch_name,
        "dataset_type": "inbox_batch",
        "batch_dir": str(batch_dir),
        "review_csv_exists": review_csv.exists(),
        "labeled_csv_exists": labeled_csv.exists(),
        "labels_json_exists": labels_json.exists(),
        "finalize_summary_exists": finalize_summary.exists(),
        "iteration_summary_exists": iteration_summary.exists(),
    }

    if labeled_csv.exists():
        with labeled_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = normalize_label(row.get("human_label", ""))
                if not label:
                    continue
                rows.append(
                    {
                        "dataset_id": batch_name,
                        "dataset_type": "inbox_batch",
                        "source_path": str(row.get("source_path", "")).strip(),
                        "human_label": label,
                        "review_rank": str(row.get("review_rank", "")).strip(),
                        "review_score": str(row.get("review_score", "")).strip(),
                        "piano_probability": str(row.get("piano_probability", "")).strip(),
                        "predicted_label": str(row.get("predicted_label", "")).strip(),
                        "label_source": "labeled_csv",
                        "source_file": str(labeled_csv),
                    }
                )
        return rows, metadata

    labels_map = safe_read_json(labels_json)
    if isinstance(labels_map, dict):
        review_rows = {}
        if review_csv.exists():
            with review_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    review_rows[str(row.get("source_path", "")).strip()] = row
        for source_path, raw_label in labels_map.items():
            label = normalize_label(raw_label)
            if not label:
                continue
            review_row = review_rows.get(str(source_path).strip(), {})
            rows.append(
                {
                    "dataset_id": batch_name,
                    "dataset_type": "inbox_batch",
                    "source_path": str(source_path).strip(),
                    "human_label": label,
                    "review_rank": str(review_row.get("review_rank", "")).strip(),
                    "review_score": str(review_row.get("review_score", "")).strip(),
                    "piano_probability": str(review_row.get("piano_probability", "")).strip(),
                    "predicted_label": str(review_row.get("predicted_label", "")).strip(),
                    "label_source": "labels_json",
                    "source_file": str(labels_json),
                }
            )
    return rows, metadata


def collect_report_rows(csv_path: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    dataset_id = csv_path.stem
    metadata = {
        "dataset_id": dataset_id,
        "dataset_type": "report_review",
        "batch_dir": "",
        "review_csv_exists": csv_path.exists(),
        "labeled_csv_exists": csv_path.exists(),
        "labels_json_exists": False,
        "finalize_summary_exists": False,
        "iteration_summary_exists": False,
    }
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = normalize_label(row.get("human_label", ""))
            if not label:
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_type": "report_review",
                    "source_path": str(row.get("source_path", row.get("path", ""))).strip(),
                    "human_label": label,
                    "review_rank": str(row.get("review_rank", "")).strip(),
                    "review_score": str(row.get("review_score", row.get("confidence_from_boundary", ""))).strip(),
                    "piano_probability": str(row.get("piano_probability", "")).strip(),
                    "predicted_label": str(row.get("predicted_label", "")).strip(),
                    "label_source": "labeled_csv",
                    "source_file": str(csv_path),
                }
            )
    return rows, metadata


def export_db_snapshots(dst_root: Path) -> dict[str, int]:
    out_counts = {"review_labels": 0, "review_events": 0}
    if not OPS_DB.exists():
        return out_counts

    conn = sqlite3.connect(str(OPS_DB))
    conn.row_factory = sqlite3.Row
    for table in ("review_labels", "review_events"):
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        out_path = dst_root / f"{table}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
        else:
            out_path.write_text("", encoding="utf-8")
        out_counts[table] = len(rows)
    return out_counts


def main():
    archive_files_root = ARCHIVE_ROOT / "files"
    snapshot_root = ARCHIVE_ROOT / "db_snapshot"
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    consolidated_rows: list[dict] = []
    inventory_rows: list[dict] = []

    for batch_dir in sorted(INBOX_ROOT.glob("unseen_batch_*")):
        rows, metadata = collect_inbox_batch_rows(batch_dir)
        consolidated_rows.extend(rows)
        metadata["label_count"] = len(rows)
        metadata["copied_review_csv"] = copy_if_exists(batch_dir / "false_positive_review.csv", archive_files_root)
        metadata["copied_labeled_csv"] = copy_if_exists(batch_dir / "false_positive_review_labeled.csv", archive_files_root)
        metadata["copied_labels_json"] = copy_if_exists(batch_dir / "false_positive_review_labels.json", archive_files_root)
        metadata["copied_finalize_summary"] = copy_if_exists(batch_dir / "finalize_summary.json", archive_files_root)
        metadata["copied_iteration_summary"] = copy_if_exists(batch_dir / "iteration_summary.json", archive_files_root)
        inventory_rows.append(metadata)

    for csv_path in sorted(REPORTS_ROOT.glob("*_labeled.csv")):
        rows, metadata = collect_report_rows(csv_path)
        consolidated_rows.extend(rows)
        metadata["label_count"] = len(rows)
        metadata["copied_labeled_csv"] = copy_if_exists(csv_path, archive_files_root)
        labels_json = csv_path.with_name(csv_path.name.replace("_labeled.csv", "_labels.json"))
        metadata["copied_labels_json"] = copy_if_exists(labels_json, archive_files_root)
        metadata["copied_review_csv"] = ""
        metadata["copied_finalize_summary"] = ""
        metadata["copied_iteration_summary"] = ""
        inventory_rows.append(metadata)

    db_counts = export_db_snapshots(snapshot_root)

    consolidated_rows.sort(key=lambda r: (r["dataset_type"], r["dataset_id"], r["source_path"]))
    inventory_rows.sort(key=lambda r: (r["dataset_type"], r["dataset_id"]))

    consolidated_csv = ARCHIVE_ROOT / "consolidated_labels.csv"
    with consolidated_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset_id",
            "dataset_type",
            "source_path",
            "human_label",
            "review_rank",
            "review_score",
            "piano_probability",
            "predicted_label",
            "label_source",
            "source_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in consolidated_rows:
            writer.writerow(row)

    inventory_csv = ARCHIVE_ROOT / "inventory.csv"
    with inventory_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset_id",
            "dataset_type",
            "batch_dir",
            "label_count",
            "review_csv_exists",
            "labeled_csv_exists",
            "labels_json_exists",
            "finalize_summary_exists",
            "iteration_summary_exists",
            "copied_review_csv",
            "copied_labeled_csv",
            "copied_labels_json",
            "copied_finalize_summary",
            "copied_iteration_summary",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in inventory_rows:
            writer.writerow(row)

    summary = {
        "archive_root": str(ARCHIVE_ROOT),
        "num_inventory_rows": len(inventory_rows),
        "num_consolidated_labels": len(consolidated_rows),
        "db_snapshot_counts": db_counts,
    }
    (ARCHIVE_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"archive_root={ARCHIVE_ROOT}")
    print(f"inventory_rows={len(inventory_rows)}")
    print(f"consolidated_labels={len(consolidated_rows)}")
    print(f"db_review_labels={db_counts['review_labels']}")
    print(f"db_review_events={db_counts['review_events']}")


if __name__ == "__main__":
    main()
