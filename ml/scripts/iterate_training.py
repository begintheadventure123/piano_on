import argparse
import hashlib
import json
import re
import sqlite3
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OPS_DB_PATH = ROOT / "ml" / "data" / "ops" / "activity.db"
ASSET_POOL_INDEX = ROOT / "ml" / "data" / "review" / "asset_pool" / "asset_index.csv"
RELABEL_EXCLUDE_PATHS = ROOT / "ml" / "data" / "review" / "asset_pool" / "relabel_exclude_paths.txt"


def run_cmd(args: list[str]) -> None:
    proc = subprocess.run(args, cwd=str(ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{details}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def ensure_columns(df: pd.DataFrame, expected: list[str], csv_path: Path) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns {missing}: {csv_path}")


def load_model_decision_threshold(model_path: Path) -> float:
    pack = joblib.load(model_path)
    if not isinstance(pack, dict):
        return 0.5
    return float(pack.get("decision_threshold", 0.5))


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_hash_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def ensure_ops_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS promoted_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sha256 TEXT NOT NULL,
                label TEXT NOT NULL,
                target_path TEXT NOT NULL,
                source_path TEXT NOT NULL,
                bucket TEXT NOT NULL,
                first_promoted_at TEXT NOT NULL,
                UNIQUE(sha256, label)
            );
            CREATE INDEX IF NOT EXISTS idx_promoted_files_label ON promoted_files(label);
            """
        )


def ensure_review_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS review_batches (
                batch_path TEXT PRIMARY KEY,
                review_csv_path TEXT NOT NULL DEFAULT '',
                model_path TEXT NOT NULL DEFAULT '',
                config_path TEXT NOT NULL DEFAULT '',
                decision_threshold REAL NOT NULL DEFAULT 0.5,
                false_positive_threshold REAL NOT NULL DEFAULT 0.5,
                total_items INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_items (
                batch_path TEXT NOT NULL,
                source_path TEXT NOT NULL,
                review_rank INTEGER NOT NULL DEFAULT 0,
                review_score REAL,
                piano_probability REAL,
                predicted_label TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(batch_path, source_path)
            );
            CREATE INDEX IF NOT EXISTS idx_review_items_batch_rank ON review_items(batch_path, review_rank);

            CREATE TABLE IF NOT EXISTS upload_file_labels (
                batch_path TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                label TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(batch_path, stored_path)
            );

            CREATE TABLE IF NOT EXISTS batch_clip_labels (
                batch_path TEXT NOT NULL,
                clip_path TEXT NOT NULL,
                label TEXT NOT NULL,
                source_file_path TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(batch_path, clip_path)
            );
            CREATE INDEX IF NOT EXISTS idx_batch_clip_labels_batch_label ON batch_clip_labels(batch_path, label);
            """
        )


def persist_review_batch(
    db_path: Path,
    batch_dir: Path,
    review_csv: Path,
    review_df: pd.DataFrame,
    *,
    model_path: Path,
    config_path: Path,
    decision_threshold: float,
    false_positive_threshold: float,
) -> None:
    ensure_review_db(db_path)
    batch_key = str(batch_dir.resolve())
    rows = review_df.fillna("").to_dict(orient="records")
    now = utc_now_iso()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO review_batches
            (batch_path, review_csv_path, model_path, config_path, decision_threshold, false_positive_threshold, total_items, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_path)
            DO UPDATE SET
                review_csv_path = excluded.review_csv_path,
                model_path = excluded.model_path,
                config_path = excluded.config_path,
                decision_threshold = excluded.decision_threshold,
                false_positive_threshold = excluded.false_positive_threshold,
                total_items = excluded.total_items,
                updated_at = excluded.updated_at
            """,
            (
                batch_key,
                str(review_csv.resolve()),
                str(model_path.resolve()),
                str(config_path.resolve()),
                float(decision_threshold),
                float(false_positive_threshold),
                int(len(rows)),
                now,
            ),
        )
        conn.execute("DELETE FROM review_items WHERE batch_path = ?", (batch_key,))
        conn.executemany(
            """
            INSERT INTO review_items
            (batch_path, source_path, review_rank, review_score, piano_probability, predicted_label)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    batch_key,
                    str(row.get("source_path", "")).strip(),
                    int(row.get("review_rank") or 0),
                    float(row.get("review_score")) if str(row.get("review_score", "")).strip() else None,
                    float(row.get("piano_probability")) if str(row.get("piano_probability", "")).strip() else None,
                    str(row.get("predicted_label", "")).strip(),
                )
                for row in rows
                if str(row.get("source_path", "")).strip()
            ],
        )


def load_upload_file_labels(db_path: Path, batch_dir: Path) -> list[dict]:
    ensure_review_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT stored_path, label
            FROM upload_file_labels
            WHERE batch_path = ?
            """,
            (str(batch_dir.resolve()),),
        ).fetchall()
    return [{"stored_path": str(r["stored_path"]), "label": str(r["label"])} for r in rows]


def persist_batch_clip_labels(db_path: Path, batch_dir: Path, clip_rows: list[dict]) -> None:
    ensure_review_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM batch_clip_labels WHERE batch_path = ?", (str(batch_dir.resolve()),))
        conn.executemany(
            """
            INSERT INTO batch_clip_labels (batch_path, clip_path, label, source_file_path, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(batch_dir.resolve()),
                    str(row["clip_path"]),
                    str(row["label"]),
                    str(row["source_file_path"]),
                    utc_now_iso(),
                )
                for row in clip_rows
            ],
        )


def apply_upload_labels_to_clips(db_path: Path, batch_dir: Path, clips_dir: Path, label_prefix: str) -> dict:
    upload_labels = load_upload_file_labels(db_path, batch_dir)
    if not upload_labels or not clips_dir.exists():
        persist_batch_clip_labels(db_path, batch_dir, [])
        return {"labeled_clips": 0, "piano": 0, "non_piano": 0}

    stem_to_label = {}
    stem_to_source = {}
    for row in upload_labels:
        raw_path = Path(str(row["stored_path"])).resolve()
        label = str(row["label"]).strip().lower()
        if label not in {"piano", "non_piano"}:
            continue
        stem_to_label[raw_path.stem.replace(" ", "_")] = label
        stem_to_source[raw_path.stem.replace(" ", "_")] = str(raw_path)

    clip_rows = []
    prefix = f"{label_prefix}_"
    for clip_path in clips_dir.rglob("*.wav"):
        clip_stem = clip_path.stem
        if not clip_stem.startswith(prefix):
            continue
        stem_body = clip_stem[len(prefix):]
        source_stem = re.sub(r"_[0-9]{5}$", "", stem_body)
        label = stem_to_label.get(source_stem, "")
        if label not in {"piano", "non_piano"}:
            continue
        clip_rows.append(
            {
                "clip_path": str(clip_path.resolve()),
                "label": label,
                "source_file_path": stem_to_source.get(source_stem, ""),
            }
        )

    persist_batch_clip_labels(db_path, batch_dir, clip_rows)
    piano_count = sum(1 for row in clip_rows if row["label"] == "piano")
    non_piano_count = sum(1 for row in clip_rows if row["label"] == "non_piano")
    return {"labeled_clips": len(clip_rows), "piano": piano_count, "non_piano": non_piano_count}


def find_promoted_path(db_path: Path, sha256: str, label: str) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT target_path FROM promoted_files WHERE sha256 = ? AND label = ?",
            (sha256, label),
        ).fetchone()
    if not row:
        return ""
    return str(row[0] or "")


def upsert_promoted_path(db_path: Path, sha256: str, label: str, target_path: Path, source_path: Path, bucket: str) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO promoted_files (sha256, label, target_path, source_path, bucket, first_promoted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256, label)
            DO UPDATE SET target_path = excluded.target_path, source_path = excluded.source_path, bucket = excluded.bucket
            """,
            (sha256, label, str(target_path), str(source_path), bucket, utc_now_iso()),
        )


def migrate_legacy_hash_index(
    db_path: Path,
    legacy_json: Path,
    label: str,
):
    legacy = load_hash_index(legacy_json)
    if not legacy:
        return
    with sqlite3.connect(str(db_path)) as conn:
        for digest, target in legacy.items():
            target_path = Path(str(target))
            conn.execute(
                """
                INSERT OR IGNORE INTO promoted_files
                (sha256, label, target_path, source_path, bucket, first_promoted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (digest, label, str(target_path), "", "", utc_now_iso()),
            )


def safe_copy_unique_name(src: Path, dst_dir: Path, bucket: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = f"{bucket}_{src.name}"
    dst = dst_dir / base
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    idx = 1
    while True:
        candidate = dst_dir / f"{bucket}_{src.stem}_{idx}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        idx += 1


def copy_with_dedup(src: Path, dst_dir: Path, bucket: str, db_path: Path, label: str) -> tuple[bool, Path | None]:
    digest = hash_file(src)
    existing = find_promoted_path(db_path=db_path, sha256=digest, label=label)
    if existing and Path(existing).exists():
        return False, Path(existing)

    copied = safe_copy_unique_name(src, dst_dir=dst_dir, bucket=bucket)
    upsert_promoted_path(
        db_path=db_path,
        sha256=digest,
        label=label,
        target_path=copied,
        source_path=src,
        bucket=bucket,
    )
    return True, copied


def build_false_positive_review(scores_csv: Path, out_csv: Path, threshold: float, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    ensure_columns(df, ["path", "piano_probability", "predicted_label"], scores_csv)
    df = df.dropna(subset=["piano_probability"]).copy()
    df["piano_probability"] = pd.to_numeric(df["piano_probability"], errors="coerce")
    df = df.dropna(subset=["piano_probability"]).copy()
    if "confidence_from_boundary" in df.columns:
        df["review_score"] = pd.to_numeric(df["confidence_from_boundary"], errors="coerce")
    else:
        df["review_score"] = (df["piano_probability"] - float(threshold)).abs()
    df = df.dropna(subset=["review_score"]).copy()
    piano_df = df[df["predicted_label"] == "piano"].copy()

    high_df = piano_df[piano_df["piano_probability"] >= threshold]
    top_df = piano_df.sort_values(by=["review_score", "piano_probability"], ascending=[False, False]).head(max(0, top_k))
    review_df = (
        pd.concat([high_df, top_df], ignore_index=True)
        .drop_duplicates(subset=["path"])
        .sort_values(by=["review_score", "piano_probability"], ascending=[False, False])
        .reset_index(drop=True)
    )

    review_df = review_df.rename(columns={"path": "source_path"})
    review_df["review_rank"] = review_df.index + 1
    review_df["review_path"] = ""
    review_df["human_label"] = ""
    review_df = review_df[
        ["review_rank", "review_score", "piano_probability", "source_path", "review_path", "human_label", "predicted_label"]
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(out_csv, index=False)
    return review_df


def build_priority_label_queue(
    scores_csv: Path,
    out_csv: Path,
    decision_threshold: float = 0.5,
    high_prob_min: float = 0.9,
    top_hard: int = 300,
    top_uncertain: int = 200,
) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    ensure_columns(df, ["path", "piano_probability", "predicted_label"], scores_csv)
    df = df.dropna(subset=["piano_probability"]).copy()
    df["piano_probability"] = pd.to_numeric(df["piano_probability"], errors="coerce")
    df = df.dropna(subset=["piano_probability"]).copy()
    df["distance_to_threshold"] = (df["piano_probability"] - float(decision_threshold)).abs()

    hard_df = df[(df["predicted_label"] == "piano") & (df["piano_probability"] >= float(high_prob_min))].copy()
    hard_df = hard_df.sort_values(by="piano_probability", ascending=False).head(max(0, int(top_hard)))
    hard_df["priority_reason"] = "hard_negative_candidate_high_prob_piano"

    uncertain_df = df.sort_values(by="distance_to_threshold", ascending=True).head(max(0, int(top_uncertain))).copy()
    uncertain_df["priority_reason"] = "near_decision_boundary"

    queue_df = (
        pd.concat([hard_df, uncertain_df], ignore_index=True)
        .drop_duplicates(subset=["path"], keep="first")
        .sort_values(by=["priority_reason", "piano_probability"], ascending=[True, False])
        .reset_index(drop=True)
    )
    queue_df["queue_rank"] = queue_df.index + 1
    cols = ["queue_rank", "priority_reason", "path", "piano_probability", "predicted_label", "distance_to_threshold"]
    queue_df = queue_df[cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    queue_df.to_csv(out_csv, index=False)
    return queue_df


def load_review_labels_from_db(db_path: Path, batch_dir: Path) -> pd.DataFrame:
    ensure_review_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            WITH manual_review AS (
                SELECT i.source_path AS source_path, l.label AS human_label, i.review_rank AS sort_rank
                FROM review_items i
                JOIN review_labels l
                  ON l.csv_path = i.batch_path
                 AND l.source_path = i.source_path
                WHERE i.batch_path = ?
            ),
            auto_clip AS (
                SELECT c.clip_path AS source_path, c.label AS human_label, 1000000 AS sort_rank
                FROM batch_clip_labels c
                WHERE c.batch_path = ?
            ),
            merged AS (
                SELECT * FROM auto_clip
                UNION ALL
                SELECT * FROM manual_review
            ),
            ranked AS (
                SELECT source_path, human_label, sort_rank,
                       ROW_NUMBER() OVER (PARTITION BY source_path ORDER BY sort_rank ASC) AS rn
                FROM merged
            )
            SELECT source_path, human_label
            FROM ranked
            WHERE rn = 1
            ORDER BY sort_rank ASC, source_path ASC
            """,
            (str(batch_dir.resolve()), str(batch_dir.resolve())),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["source_path", "human_label"])
    return pd.DataFrame([{"source_path": str(r["source_path"]), "human_label": str(r["human_label"])} for r in rows])


def promote_labels_df(
    df: pd.DataFrame,
    bucket: str,
    non_piano_target_root: Path,
    piano_target_root: Path,
) -> dict:
    ensure_columns(df, ["source_path", "human_label"], Path("<db_or_csv>"))

    non_piano_root = non_piano_target_root / bucket
    piano_root = piano_target_root / bucket
    db_path = OPS_DB_PATH
    ensure_ops_db(db_path)
    migrate_legacy_hash_index(db_path, non_piano_target_root / ".dedup_hash_index.json", "non_piano")
    migrate_legacy_hash_index(db_path, piano_target_root / ".dedup_hash_index.json", "piano")

    copied_non_piano = 0
    copied_piano = 0
    dedup_skipped_non_piano = 0
    dedup_skipped_piano = 0
    skipped = 0
    relabel_excluded_paths: set[str] = set()

    asset_index_by_managed: dict[str, dict] = {}
    if ASSET_POOL_INDEX.exists():
        asset_df = pd.read_csv(ASSET_POOL_INDEX).fillna("")
        for asset_row in asset_df.to_dict(orient="records"):
            managed_path = str(asset_row.get("managed_path", "")).strip()
            if managed_path:
                asset_index_by_managed[str(Path(managed_path).resolve())] = asset_row

    for row in df.itertuples(index=False):
        src = Path(str(row.source_path))
        label = str(row.human_label).strip().lower()
        if label not in {"non_piano", "piano"}:
            skipped += 1
            continue
        if not src.exists():
            skipped += 1
            continue

        asset_meta = asset_index_by_managed.get(str(src.resolve()))
        if asset_meta:
            original_label = str(asset_meta.get("human_label", "")).strip().lower()
            if original_label and original_label != label:
                for key in ("canonical_source_path", "promoted_path"):
                    raw = str(asset_meta.get(key, "")).strip()
                    if raw:
                        path = Path(raw)
                        if path.exists():
                            relabel_excluded_paths.add(str(path.resolve()))

        if label == "non_piano":
            copied, _ = copy_with_dedup(
                src=src,
                dst_dir=non_piano_root,
                bucket=bucket,
                db_path=db_path,
                label="non_piano",
            )
            if copied:
                copied_non_piano += 1
            else:
                dedup_skipped_non_piano += 1
        else:
            copied, _ = copy_with_dedup(
                src=src,
                dst_dir=piano_root,
                bucket=bucket,
                db_path=db_path,
                label="piano",
            )
            if copied:
                copied_piano += 1
            else:
                dedup_skipped_piano += 1

    if relabel_excluded_paths:
        RELABEL_EXCLUDE_PATHS.parent.mkdir(parents=True, exist_ok=True)
        existing = set()
        if RELABEL_EXCLUDE_PATHS.exists():
            existing = {
                str(Path(line.strip()).resolve())
                for line in RELABEL_EXCLUDE_PATHS.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        merged = sorted(existing | relabel_excluded_paths)
        RELABEL_EXCLUDE_PATHS.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

    return {
        "copied_non_piano": copied_non_piano,
        "copied_piano": copied_piano,
        "dedup_skipped_non_piano": dedup_skipped_non_piano,
        "dedup_skipped_piano": dedup_skipped_piano,
        "skipped": skipped,
        "relabel_excluded_paths": len(relabel_excluded_paths),
        "non_piano_target": str(non_piano_root),
        "piano_target": str(piano_root),
        "activity_db": str(db_path),
    }


def promote_labels(
    labeled_csv: Path,
    bucket: str,
    non_piano_target_root: Path,
    piano_target_root: Path,
) -> dict:
    if not labeled_csv.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {labeled_csv}")
    df = pd.read_csv(labeled_csv)
    return promote_labels_df(
        df=df,
        bucket=bucket,
        non_piano_target_root=non_piano_target_root,
        piano_target_root=piano_target_root,
    )


def retrain_model(
    raw_root: Path,
    manifest_out: Path,
    config: Path,
    model_out: Path,
    report_out: Path,
    train_script_name: str = "train_baseline.py",
) -> None:
    prepare_script = ROOT / "ml" / "scripts" / "prepare_manifest.py"
    train_script = ROOT / "ml" / "scripts" / train_script_name

    run_cmd(
        [
            sys.executable,
            str(prepare_script),
            "--root",
            str(raw_root.resolve()),
            "--out",
            str(manifest_out.resolve()),
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(train_script),
            "--manifest",
            str(manifest_out.resolve()),
            "--config",
            str(config.resolve()),
            "--model-out",
            str(model_out.resolve()),
            "--report-out",
            str(report_out.resolve()),
        ]
    )


def sync_labeled_asset_pool() -> None:
    sync_script = ROOT / "ml" / "scripts" / "audit_labeled_assets.py"
    materialize_script = ROOT / "ml" / "scripts" / "sync_labeled_asset_pool.py"
    run_cmd([sys.executable, str(sync_script)])
    run_cmd([sys.executable, str(materialize_script)])


def cleanup_batch_artifacts(batch_dir: Path) -> dict:
    removed_dirs = []
    removed_files = []

    for d in ("raw_long", "clips"):
        p = batch_dir / d
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            removed_dirs.append(str(p))

    # Remove generated heavy scores; keep labeled csv/json + summary for audit.
    file_globs = [
        "clip_scores*.csv",
        "clip_low_confidence_review*.csv",
    ]
    for pat in file_globs:
        for p in batch_dir.glob(pat):
            if p.exists() and p.is_file():
                p.unlink()
                removed_files.append(str(p))

    return {"removed_dirs": removed_dirs, "removed_files": removed_files}


def cmd_run_batch(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch).resolve()
    raw_dir = batch_dir / "raw_long"
    clips_dir = batch_dir / "clips"
    scores_csv = batch_dir / args.scores_csv
    review_csv = batch_dir / args.review_csv
    model_path = Path(args.model).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw_long folder: {raw_dir}")

    decision_threshold = load_model_decision_threshold(model_path)
    false_positive_threshold = (
        float(args.false_positive_threshold) if args.false_positive_threshold is not None else decision_threshold
    )

    split_script = ROOT / "ml" / "scripts" / "split_long_audio.py"
    score_script = ROOT / "ml" / "scripts" / "score_clips.py"
    run_cmd(
        [
            sys.executable,
            str(split_script),
            "--input",
            str(raw_dir),
            "--out",
            str(clips_dir),
            "--label-prefix",
            args.label_prefix or batch_dir.name,
            "--clip-seconds",
            str(args.clip_seconds),
            "--hop-seconds",
            str(args.hop_seconds),
            "--min-rms-db",
            str(args.min_rms_db),
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(score_script),
            "--input",
            str(clips_dir),
            "--model",
            str(model_path),
            "--config",
            str(Path(args.config).resolve()),
            "--out-csv",
            str(scores_csv),
            "--review-csv",
            str(batch_dir / "clip_low_confidence_review.csv"),
        ]
    )

    effective_label_prefix = args.label_prefix or batch_dir.name
    auto_clip_label_summary = apply_upload_labels_to_clips(
        db_path=OPS_DB_PATH,
        batch_dir=batch_dir,
        clips_dir=clips_dir,
        label_prefix=effective_label_prefix,
    )

    review_df = build_false_positive_review(
        scores_csv=scores_csv,
        out_csv=review_csv,
        threshold=false_positive_threshold,
        top_k=args.top_k,
    )
    priority_queue_csv = batch_dir / "priority_label_queue.csv"
    priority_df = build_priority_label_queue(
        scores_csv=scores_csv,
        out_csv=priority_queue_csv,
        decision_threshold=decision_threshold,
        high_prob_min=0.9,
        top_hard=300,
        top_uncertain=200,
    )

    score_df = pd.read_csv(scores_csv)
    predicted_piano = int((score_df["predicted_label"] == "piano").sum())
    summary = {
        "batch_dir": str(batch_dir),
        "scores_csv": str(scores_csv),
        "review_csv": str(review_csv),
        "review_store": str(OPS_DB_PATH),
        "total_scored": int(len(score_df)),
        "predicted_piano": predicted_piano,
        "predicted_non_piano": int((score_df["predicted_label"] == "non_piano").sum()),
        "false_positive_candidates": int(len(review_df)),
        "priority_queue_csv": str(priority_queue_csv),
        "priority_queue_size": int(len(priority_df)),
        "decision_threshold": float(decision_threshold),
        "false_positive_threshold": float(false_positive_threshold),
        "top_k": int(args.top_k),
        "auto_clip_labels": auto_clip_label_summary,
    }

    if args.auto_promote_all_non_piano:
        target_root = Path(args.non_piano_target).resolve() / (args.bucket or batch_dir.name)
        non_piano_target_root = Path(args.non_piano_target).resolve()
        db_path = OPS_DB_PATH
        ensure_ops_db(db_path)
        migrate_legacy_hash_index(db_path, non_piano_target_root / ".dedup_hash_index.json", "non_piano")
        copied = 0
        dedup_skipped = 0
        for p in clips_dir.rglob("*"):
            if p.is_file():
                did_copy, _ = copy_with_dedup(
                    src=p,
                    dst_dir=target_root,
                    bucket=batch_dir.name,
                    db_path=db_path,
                    label="non_piano",
                )
                if did_copy:
                    copied += 1
                else:
                    dedup_skipped += 1
        summary["auto_promoted_non_piano"] = copied
        summary["auto_promote_dedup_skipped"] = dedup_skipped
        summary["auto_promoted_to"] = str(target_root)
        summary["activity_db"] = str(db_path)

    persist_review_batch(
        db_path=OPS_DB_PATH,
        batch_dir=batch_dir,
        review_csv=review_csv,
        review_df=review_df,
        model_path=model_path,
        config_path=Path(args.config).resolve(),
        decision_threshold=decision_threshold,
        false_positive_threshold=false_positive_threshold,
    )

    summary_path = batch_dir / "iteration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_json={summary_path}")
    print("launch_reviewer=python ml/tools/clip_reviewer/server.py --root .")


def cmd_promote_labels(args: argparse.Namespace) -> None:
    if args.batch:
        batch_dir = Path(args.batch).resolve()
        bucket = args.bucket or batch_dir.name
        df = load_review_labels_from_db(OPS_DB_PATH, batch_dir=batch_dir)
        result = promote_labels_df(
            df=df,
            bucket=bucket,
            non_piano_target_root=Path(args.non_piano_target).resolve(),
            piano_target_root=Path(args.piano_target).resolve(),
        )
    elif args.labeled_csv:
        labeled_csv = Path(args.labeled_csv).resolve()
        bucket = args.bucket or labeled_csv.stem
        result = promote_labels(
            labeled_csv=labeled_csv,
            bucket=bucket,
            non_piano_target_root=Path(args.non_piano_target).resolve(),
            piano_target_root=Path(args.piano_target).resolve(),
        )
    else:
        raise ValueError("Either --batch or --labeled-csv is required")
    sync_labeled_asset_pool()
    for k, v in result.items():
        print(f"{k}={v}")


def cmd_retrain(args: argparse.Namespace) -> None:
    retrain_model(
        raw_root=Path(args.raw_root),
        manifest_out=Path(args.manifest_out),
        config=Path(args.config),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
    )


def cmd_retrain_v2(args: argparse.Namespace) -> None:
    retrain_model(
        raw_root=Path(args.raw_root),
        manifest_out=Path(args.manifest_out),
        config=Path(args.config),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
        train_script_name="train_v2.py",
    )


def cmd_retrain_v3(args: argparse.Namespace) -> None:
    retrain_model(
        raw_root=Path(args.raw_root),
        manifest_out=Path(args.manifest_out),
        config=Path(args.config),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
        train_script_name="train_v3.py",
    )


def cmd_build_ensemble_v23(args: argparse.Namespace) -> None:
    script_path = ROOT / "ml" / "scripts" / "build_ensemble_v23.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--v2-model",
        str(Path(args.v2_model).resolve()),
        "--v2-config",
        str(Path(args.v2_config).resolve()),
        "--v3-model",
        str(Path(args.v3_model).resolve()),
        "--v3-config",
        str(Path(args.v3_config).resolve()),
        "--out-model",
        str(Path(args.model_out).resolve()),
        "--out-report",
        str(Path(args.report_out).resolve()),
        "--weight-v2",
        str(args.weight_v2),
        "--weight-v3",
        str(args.weight_v3),
        "--threshold",
        str(args.threshold),
        "--review-threshold",
        str(args.review_threshold),
        "--method",
        str(args.method),
    ]
    run_cmd(cmd)


def cmd_finalize_batch(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch).resolve()
    bucket = args.bucket or batch_dir.name
    t0 = time.time()
    if args.labeled_csv:
        labeled_csv = Path(args.labeled_csv).resolve()
        promoted = promote_labels(
            labeled_csv=labeled_csv,
            bucket=bucket,
            non_piano_target_root=Path(args.non_piano_target).resolve(),
            piano_target_root=Path(args.piano_target).resolve(),
        )
        labeled_source = str(labeled_csv)
    else:
        df = load_review_labels_from_db(OPS_DB_PATH, batch_dir=batch_dir)
        promoted = promote_labels_df(
            df=df,
            bucket=bucket,
            non_piano_target_root=Path(args.non_piano_target).resolve(),
            piano_target_root=Path(args.piano_target).resolve(),
        )
        labeled_source = f"db:{batch_dir}"
    sync_labeled_asset_pool()
    retrain_model(
        raw_root=Path(args.raw_root),
        manifest_out=Path(args.manifest_out),
        config=Path(args.config),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
        train_script_name=args.train_script,
    )
    cleanup = cleanup_batch_artifacts(batch_dir=batch_dir) if args.cleanup_batch else {"removed_dirs": [], "removed_files": []}

    summary = {
        "batch_dir": str(batch_dir),
        "labeled_source": labeled_source,
        "bucket": bucket,
        "promote": promoted,
        "cleanup": cleanup,
        "duration_seconds": round(time.time() - t0, 2),
        "completed_at_epoch": int(time.time()),
    }
    out = batch_dir / "finalize_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"finalize_summary={out}")


def cmd_cleanup_retention(args: argparse.Namespace) -> None:
    inbox_root = Path(args.inbox_root).resolve()
    if not inbox_root.exists():
        print(f"inbox_root_missing={inbox_root}")
        return
    cutoff = time.time() - (args.ttl_days * 86400.0)
    batches_scanned = 0
    batches_matched = 0
    batches_deleted = 0
    dirs_removed = 0
    files_removed = 0

    for batch_dir in inbox_root.iterdir():
        if not batch_dir.is_dir():
            continue
        batches_scanned += 1
        mtime = batch_dir.stat().st_mtime
        if mtime > cutoff:
            continue
        batches_matched += 1
        if args.delete_batch_completely:
            shutil.rmtree(batch_dir)
            batches_deleted += 1
            continue

        cleanup = cleanup_batch_artifacts(batch_dir=batch_dir)
        dirs_removed += len(cleanup["removed_dirs"])
        files_removed += len(cleanup["removed_files"])

    print(f"inbox_root={inbox_root}")
    print(f"ttl_days={args.ttl_days}")
    print(f"batches_scanned={batches_scanned}")
    print(f"batches_matched={batches_matched}")
    print(f"batches_deleted={batches_deleted}")
    print(f"dirs_removed={dirs_removed}")
    print(f"files_removed={files_removed}")


def cmd_eval_holdout(args: argparse.Namespace) -> None:
    eval_script = ROOT / "ml" / "scripts" / "evaluate_holdout.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--holdout-root",
        str(args.holdout_root),
        "--model",
        str(args.model),
        "--config",
        str(args.config),
        "--out-report",
        str(args.out_report),
        "--out-preds",
        str(args.out_preds),
    ]
    run_cmd(cmd)


def cmd_audit_dataset(args: argparse.Namespace) -> None:
    audit_script = ROOT / "ml" / "scripts" / "audit_dataset.py"
    cmd = [
        sys.executable,
        str(audit_script),
        "--manifest",
        str(args.manifest),
        "--out-json",
        str(args.out_json),
        "--out-suspects-csv",
        str(args.out_suspects_csv),
        "--out-conflicts-csv",
        str(args.out_conflicts_csv),
    ]
    run_cmd(cmd)


def cmd_combine_scores(args: argparse.Namespace) -> None:
    combine_script = ROOT / "ml" / "scripts" / "combine_model_scores.py"
    cmd = [
        sys.executable,
        str(combine_script),
        "--scores",
        str(args.scores),
        "--weights",
        str(args.weights),
        "--threshold",
        str(args.threshold),
        "--out-csv",
        str(args.out_csv),
    ]
    run_cmd(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage iterative false-positive hard-negative training loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_batch = sub.add_parser("run-batch", help="Split+score one inbox batch and persist review queue")
    run_batch.add_argument("--batch", required=True, help="Batch folder (expects raw_long under it)")
    run_batch.add_argument("--model", default="ml/models/baseline_logreg.joblib")
    run_batch.add_argument("--config", default="ml/configs/train_config.yaml")
    run_batch.add_argument("--label-prefix", default="")
    run_batch.add_argument("--clip-seconds", type=float, default=8.0)
    run_batch.add_argument("--hop-seconds", type=float, default=8.0)
    run_batch.add_argument("--min-rms-db", type=float, default=-50.0)
    run_batch.add_argument("--scores-csv", default="clip_scores.csv")
    run_batch.add_argument("--review-csv", default="false_positive_review.csv")
    run_batch.add_argument("--false-positive-threshold", type=float, default=None)
    run_batch.add_argument("--top-k", type=int, default=120)
    run_batch.add_argument("--auto-promote-all-non-piano", action="store_true")
    run_batch.add_argument("--non-piano-target", default="ml/data/raw/non_piano/hard_negative")
    run_batch.add_argument("--bucket", default="")
    run_batch.set_defaults(func=cmd_run_batch)

    promote = sub.add_parser("promote-labels", help="Copy reviewed labels into training folders (dedup enabled)")
    promote.add_argument("--batch", default="", help="Batch folder to promote labels from DB-backed review state")
    promote.add_argument("--labeled-csv", default="", help="Legacy CSV export from clip reviewer")
    promote.add_argument("--bucket", default="", help="Optional dataset bucket name")
    promote.add_argument("--non-piano-target", default="ml/data/raw/non_piano/hard_negative")
    promote.add_argument("--piano-target", default="ml/data/raw/piano/hard_positive")
    promote.set_defaults(func=cmd_promote_labels)

    retrain = sub.add_parser("retrain", help="Prepare manifest and retrain baseline model")
    retrain.add_argument("--raw-root", default="ml/data/raw")
    retrain.add_argument("--manifest-out", default="ml/data/labels/manifest.csv")
    retrain.add_argument("--config", default="ml/configs/train_config.yaml")
    retrain.add_argument("--model-out", default="ml/models/baseline_logreg.joblib")
    retrain.add_argument("--report-out", default="ml/reports/baseline_metrics.json")
    retrain.set_defaults(func=cmd_retrain)

    retrain_v2 = sub.add_parser("retrain-v2", help="Prepare manifest and retrain the richer-feature v2 model")
    retrain_v2.add_argument("--raw-root", default="ml/data/raw")
    retrain_v2.add_argument("--manifest-out", default="ml/data/labels/manifest.csv")
    retrain_v2.add_argument("--config", default="ml/configs/train_config_v2.yaml")
    retrain_v2.add_argument("--model-out", default="ml/models/piano_detector_v2.joblib")
    retrain_v2.add_argument("--report-out", default="ml/reports/piano_detector_v2_metrics.json")
    retrain_v2.set_defaults(func=cmd_retrain_v2)

    retrain_v3 = sub.add_parser("retrain-v3", help="Prepare manifest and retrain the temporal v3 model")
    retrain_v3.add_argument("--raw-root", default="ml/data/raw")
    retrain_v3.add_argument("--manifest-out", default="ml/data/labels/manifest.csv")
    retrain_v3.add_argument("--config", default="ml/configs/train_config_v3.yaml")
    retrain_v3.add_argument("--model-out", default="ml/models/piano_detector_v3.joblib")
    retrain_v3.add_argument("--report-out", default="ml/reports/piano_detector_v3_metrics.json")
    retrain_v3.set_defaults(func=cmd_retrain_v3)

    ensemble_v23 = sub.add_parser("build-ensemble-v23", help="Build weighted v2+v3 ensemble pack")
    ensemble_v23.add_argument("--v2-model", default="ml/models/piano_detector_v2.joblib")
    ensemble_v23.add_argument("--v2-config", default="ml/configs/train_config_v2.yaml")
    ensemble_v23.add_argument("--v3-model", default="ml/models/piano_detector_v3.joblib")
    ensemble_v23.add_argument("--v3-config", default="ml/configs/train_config_v3.yaml")
    ensemble_v23.add_argument("--model-out", default="ml/models/piano_detector_v23_ensemble.joblib")
    ensemble_v23.add_argument("--report-out", default="ml/reports/piano_detector_v23_ensemble_build.json")
    ensemble_v23.add_argument("--weight-v2", type=float, default=0.7)
    ensemble_v23.add_argument("--weight-v3", type=float, default=0.3)
    ensemble_v23.add_argument("--threshold", type=float, default=0.84)
    ensemble_v23.add_argument("--review-threshold", type=float, default=0.72)
    ensemble_v23.add_argument("--method", default="weighted_average", choices=["weighted_average", "min", "max"])
    ensemble_v23.set_defaults(func=cmd_build_ensemble_v23)

    finalize = sub.add_parser(
        "finalize-batch",
        help="Promote labels + retrain + cleanup batch artifacts (recommended end-of-batch flow)",
    )
    finalize.add_argument("--batch", required=True)
    finalize.add_argument("--labeled-csv", default="", help="Legacy fallback. Default path is DB-backed batch labels.")
    finalize.add_argument("--bucket", default="")
    finalize.add_argument("--non-piano-target", default="ml/data/raw/non_piano/hard_negative")
    finalize.add_argument("--piano-target", default="ml/data/raw/piano/hard_positive")
    finalize.add_argument("--raw-root", default="ml/data/raw")
    finalize.add_argument("--manifest-out", default="ml/data/labels/manifest.csv")
    finalize.add_argument("--config", default="ml/configs/train_config.yaml")
    finalize.add_argument("--model-out", default="ml/models/baseline_logreg.joblib")
    finalize.add_argument("--report-out", default="ml/reports/baseline_metrics.json")
    finalize.add_argument("--train-script", default="train_baseline.py")
    finalize.add_argument("--cleanup-batch", action="store_true", default=True)
    finalize.set_defaults(func=cmd_finalize_batch)

    cleanup = sub.add_parser(
        "cleanup-retention",
        help="Apply inbox retention policy. Default: strip raw/clip artifacts for batches older than TTL.",
    )
    cleanup.add_argument("--inbox-root", default="ml/data/inbox")
    cleanup.add_argument("--ttl-days", type=int, default=14)
    cleanup.add_argument("--delete-batch-completely", action="store_true")
    cleanup.set_defaults(func=cmd_cleanup_retention)

    holdout = sub.add_parser("eval-holdout", help="Evaluate model on fixed holdout set")
    holdout.add_argument("--holdout-root", default="ml/data/eval/holdout")
    holdout.add_argument("--model", default="ml/models/baseline_logreg.joblib")
    holdout.add_argument("--config", default="ml/configs/train_config.yaml")
    holdout.add_argument("--out-report", default="ml/reports/holdout_metrics.json")
    holdout.add_argument("--out-preds", default="ml/reports/holdout_predictions.csv")
    holdout.set_defaults(func=cmd_eval_holdout)

    audit = sub.add_parser("audit-dataset", help="Audit manifest for label and distribution risks")
    audit.add_argument("--manifest", default="ml/data/labels/manifest.csv")
    audit.add_argument("--out-json", default="ml/reports/dataset_audit.json")
    audit.add_argument("--out-suspects-csv", default="ml/reports/dataset_suspects.csv")
    audit.add_argument("--out-conflicts-csv", default="ml/reports/dataset_conflicts.csv")
    audit.set_defaults(func=cmd_audit_dataset)

    combine = sub.add_parser("combine-scores", help="Combine probability outputs from multiple model score CSVs")
    combine.add_argument("--scores", required=True, help="Comma-separated score CSV paths")
    combine.add_argument("--weights", default="", help="Comma-separated weights aligned with --scores")
    combine.add_argument("--threshold", type=float, default=0.5)
    combine.add_argument("--out-csv", default="ml/reports/ensemble_clip_scores.csv")
    combine.set_defaults(func=cmd_combine_scores)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
