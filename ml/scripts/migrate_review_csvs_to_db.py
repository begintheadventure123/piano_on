import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.tools.clip_reviewer.server import ActivityDb, VALID_LABELS


INBOX_ROOT = ROOT / "ml" / "data" / "inbox"
OPS_DB_PATH = ROOT / "ml" / "data" / "ops" / "activity.db"


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        if "source_path" not in row and "path" in row:
            row["source_path"] = row["path"]
    return rows


def find_review_csvs() -> list[Path]:
    preferred: dict[Path, Path] = {}
    for csv_path in INBOX_ROOT.rglob("false_positive_review*.csv"):
        batch_dir = csv_path.parent
        current = preferred.get(batch_dir)
        if current is None:
            preferred[batch_dir] = csv_path
            continue
        if current.name.endswith("_labeled.csv") and csv_path.name == "false_positive_review.csv":
            preferred[batch_dir] = csv_path
    return sorted(preferred.values())


def main() -> None:
    db = ActivityDb(OPS_DB_PATH)
    migrated_batches = 0
    migrated_items = 0
    seeded_labels = 0

    for csv_path in find_review_csvs():
        batch_dir = csv_path.parent.resolve()
        rows = load_rows(csv_path)
        if not rows:
            continue
        db.replace_review_batch(batch_dir, rows, review_csv_path=csv_path)
        migrated_batches += 1
        migrated_items += len(rows)
        for row in rows:
            label = str(row.get("human_label", "")).strip().lower()
            source_path = str(row.get("source_path", "")).strip()
            if not source_path or label not in VALID_LABELS or not label:
                continue
            db.set_label(str(batch_dir), source_path, label, str(batch_dir))
            seeded_labels += 1

    print(f"db_path={OPS_DB_PATH}")
    print(f"migrated_batches={migrated_batches}")
    print(f"migrated_items={migrated_items}")
    print(f"seeded_labels={seeded_labels}")


if __name__ == "__main__":
    main()
