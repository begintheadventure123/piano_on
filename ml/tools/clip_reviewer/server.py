import argparse
import cgi
import csv
from datetime import datetime
import hashlib
import json
import mimetypes
import os
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


VALID_LABELS = {"piano", "non_piano", "unclear", ""}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".wma", ".flac", ".aiff", ".aif"}
LONG_AUDIO_LABELS = {"piano", "non_piano", "unclear", ""}
STATUS_SUCCESS = "success"
STATUS_DUPLICATE = "duplicate"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class ActivityDb:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sha256 TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    original_name TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    batch_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    stored_path TEXT,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(file_id) REFERENCES files(id)
                );

                CREATE INDEX IF NOT EXISTS idx_uploads_status_time ON uploads(status, uploaded_at DESC);
                CREATE INDEX IF NOT EXISTS idx_uploads_batch_time ON uploads(batch_path, uploaded_at DESC);

                CREATE TABLE IF NOT EXISTS review_labels (
                    csv_path TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(csv_path, source_path)
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    csv_path TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    active_batch TEXT NOT NULL DEFAULT '',
                    event_at TEXT NOT NULL
                );

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

    def upsert_file(self, sha256: str, size_bytes: int, original_name: str) -> int:
        now = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM files WHERE sha256 = ?", (sha256,)).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO files (sha256, size_bytes, original_name, first_seen_at)
                VALUES (?, ?, ?, ?)
                """,
                (sha256, int(size_bytes), original_name, now),
            )
            return int(cur.lastrowid)

    def find_success_by_sha(self, sha256: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.stored_path, f.id AS file_id, f.size_bytes
                FROM files f
                JOIN uploads u ON u.file_id = f.id
                WHERE f.sha256 = ? AND u.status = ?
                ORDER BY u.id DESC
                LIMIT 1
                """,
                (sha256, STATUS_SUCCESS),
            ).fetchone()
            if not row:
                return None
            return {
                "stored_path": str(row["stored_path"] or ""),
                "file_id": int(row["file_id"]),
                "size_bytes": int(row["size_bytes"]),
            }

    def log_upload(
        self,
        file_id: int | None,
        batch_path: Path,
        source_name: str,
        stored_path: str,
        status: str,
        reason: str = "",
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO uploads (file_id, batch_path, source_name, stored_path, uploaded_at, status, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    str(batch_path),
                    source_name,
                    stored_path,
                    utc_now_iso(),
                    status,
                    reason,
                ),
            )

    def get_upload_history(self, min_size_bytes: int, limit: int, statuses: list[str]) -> list[dict]:
        if not statuses:
            statuses = [STATUS_SUCCESS]
        placeholders = ",".join("?" for _ in statuses)
        query = f"""
            SELECT u.id, u.uploaded_at, u.status, u.reason, u.batch_path, u.source_name, u.stored_path,
                   f.sha256, f.size_bytes
            FROM uploads u
            LEFT JOIN files f ON f.id = u.file_id
            WHERE u.status IN ({placeholders})
              AND COALESCE(f.size_bytes, 0) >= ?
            ORDER BY u.id DESC
            LIMIT ?
        """
        params = [*statuses, int(min_size_bytes), int(limit)]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "uploaded_at": str(row["uploaded_at"]),
                    "status": str(row["status"]),
                    "reason": str(row["reason"] or ""),
                    "batch_path": str(row["batch_path"]),
                    "source_name": str(row["source_name"]),
                    "stored_path": str(row["stored_path"] or ""),
                    "sha256": str(row["sha256"] or ""),
                    "size_bytes": int(row["size_bytes"] or 0),
                }
            )
        return out

    def load_labels(self, review_key: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_path, label FROM review_labels WHERE csv_path = ?",
                (str(review_key),),
            ).fetchall()
        return {str(r["source_path"]): str(r["label"]) for r in rows if str(r["label"]) in VALID_LABELS and str(r["label"])}

    def set_label(self, review_key: str, source_path: str, label: str, active_batch: str):
        now = utc_now_iso()
        with self._connect() as conn:
            if label:
                conn.execute(
                    """
                    INSERT INTO review_labels (csv_path, source_path, label, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(csv_path, source_path)
                    DO UPDATE SET label = excluded.label, updated_at = excluded.updated_at
                    """,
                    (str(review_key), source_path, label, now),
                )
            else:
                conn.execute(
                    "DELETE FROM review_labels WHERE csv_path = ? AND source_path = ?",
                    (str(review_key), source_path),
                )
            conn.execute(
                """
                INSERT INTO review_events (csv_path, source_path, label, active_batch, event_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(review_key), source_path, label, active_batch, now),
            )

    def overview(self) -> dict:
        with self._connect() as conn:
            total_files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
            total_uploads = int(conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0])
            duplicate_uploads = int(
                conn.execute("SELECT COUNT(*) FROM uploads WHERE status = ?", (STATUS_DUPLICATE,)).fetchone()[0]
            )
            success_uploads = int(
                conn.execute("SELECT COUNT(*) FROM uploads WHERE status = ?", (STATUS_SUCCESS,)).fetchone()[0]
            )
            review_labels = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM review_labels l
                    WHERE EXISTS (
                        SELECT 1
                        FROM review_batches b
                        WHERE b.batch_path = l.csv_path
                    )
                    """
                ).fetchone()[0]
            )
            if review_labels == 0:
                review_labels = int(conn.execute("SELECT COUNT(*) FROM review_labels").fetchone()[0])
        return {
            "db_path": str(self.db_path),
            "total_files": total_files,
            "total_uploads": total_uploads,
            "success_uploads": success_uploads,
            "duplicate_uploads": duplicate_uploads,
            "review_labels": review_labels,
        }

    def most_recent_review_csv(self) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT csv_path
                FROM review_labels
                GROUP BY csv_path
                ORDER BY MAX(updated_at) DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        csv_path = Path(str(row["csv_path"])).resolve()
        return csv_path if csv_path.exists() else None

    def most_recent_review_batch(self) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT batch_path
                FROM review_batches
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        batch_path = Path(str(row["batch_path"])).resolve()
        return batch_path if batch_path.exists() else None

    def load_review_items(self, batch_path: Path) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_path, review_rank, review_score, piano_probability, predicted_label
                FROM review_items
                WHERE batch_path = ?
                ORDER BY review_rank ASC, source_path ASC
                """,
                (str(batch_path.resolve()),),
            ).fetchall()
        return [dict(row) for row in rows]

    def known_review_batch(self, batch_path: Path) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM review_batches WHERE batch_path = ? LIMIT 1",
                (str(batch_path.resolve()),),
            ).fetchone()
        return row is not None

    def set_upload_file_label(self, batch_path: Path, stored_path: Path, label: str) -> None:
        if label not in {"piano", "non_piano"}:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO upload_file_labels (batch_path, stored_path, label, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_path, stored_path)
                DO UPDATE SET label = excluded.label, updated_at = excluded.updated_at
                """,
                (str(batch_path.resolve()), str(stored_path.resolve()), label, utc_now_iso()),
            )

    def load_auto_clip_labels(self, batch_path: Path) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT clip_path, label FROM batch_clip_labels WHERE batch_path = ?",
                (str(batch_path.resolve()),),
            ).fetchall()
        return {
            str(row["clip_path"]): str(row["label"])
            for row in rows
            if str(row["label"]) in {"piano", "non_piano"}
        }

    def replace_review_batch(self, batch_path: Path, rows: list[dict], review_csv_path: Path | None = None) -> None:
        batch_key = str(batch_path.resolve())
        csv_value = str(review_csv_path.resolve()) if review_csv_path else ""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_batches
                (batch_path, review_csv_path, total_items, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_path)
                DO UPDATE SET
                    review_csv_path = CASE
                        WHEN excluded.review_csv_path <> '' THEN excluded.review_csv_path
                        ELSE review_batches.review_csv_path
                    END,
                    total_items = excluded.total_items,
                    updated_at = excluded.updated_at
                """,
                (batch_key, csv_value, int(len(rows)), now),
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


def copy_stream_to_file_with_hash(stream, target: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    with target.open("wb") as out_f:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
            h.update(chunk)
            total += len(chunk)
    return total, h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Central dashboard for iterative model improvement")
    parser.add_argument("--csv", default="", help="Optional initial review CSV")
    parser.add_argument("--root", default=".", help="Workspace root to serve static files from")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    return parser.parse_args()


def run_command(args: list[str], cwd: Path) -> dict:
    proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "command": " ".join(args),
    }


def run_command_cancellable(
    args: list[str],
    cwd: Path,
    cancel_event: threading.Event,
    proc_lock: threading.Lock,
    proc_slot: dict,
) -> dict:
    proc = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with proc_lock:
        proc_slot["proc"] = proc
    try:
        while True:
            if cancel_event.is_set():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                stdout, stderr = proc.communicate()
                return {
                    "returncode": -1,
                    "stdout": (stdout or "").strip(),
                    "stderr": ((stderr or "").strip() + "\nCancelled by user.").strip(),
                    "command": " ".join(args),
                    "cancelled": True,
                }
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                return {
                    "returncode": proc.returncode,
                    "stdout": (stdout or "").strip(),
                    "stderr": (stderr or "").strip(),
                    "command": " ".join(args),
                    "cancelled": False,
                }
            time.sleep(0.2)
    finally:
        with proc_lock:
            proc_slot["proc"] = None


@dataclass
class ReviewContext:
    batch_path: Path
    review_csv_path: Path
    export_out: Path


def infer_context_for_batch(batch_path: Path) -> ReviewContext:
    review_csv_path = batch_path / "false_positive_review.csv"
    return ReviewContext(
        batch_path=batch_path,
        review_csv_path=review_csv_path,
        export_out=batch_path / "review_labels_export.csv",
    )


class ReviewState:
    def __init__(self, root_dir: Path, initial_csv: Path | None, db: ActivityDb):
        self.root_dir = root_dir
        self.db = db
        self.context: ReviewContext | None = None
        self.active_batch: Path | None = None
        self.rows: list[dict] = []
        self.auto_labels: dict[str, str] = {}
        self.labels: dict[str, str] = {}
        if initial_csv:
            self.load_csv(initial_csv)
        else:
            latest_batch = self._latest_inbox_review_batch()
            if latest_batch:
                self.load_batch(latest_batch)
            else:
                most_recent_batch = self.db.most_recent_review_batch()
                if most_recent_batch:
                    self.load_batch(most_recent_batch)
                else:
                    latest_batch_csv = self._latest_inbox_review_csv()
                    if latest_batch_csv:
                        self.load_csv(latest_batch_csv)
                        return
                    most_recent_csv = self.db.most_recent_review_csv()
                    if most_recent_csv:
                        self.load_csv(most_recent_csv)

    def _latest_inbox_review_batch(self) -> Path | None:
        inbox_root = (self.root_dir / "ml" / "data" / "inbox").resolve()
        if not inbox_root.exists():
            return None
        candidates = []
        for p in inbox_root.iterdir():
            if not p.is_dir():
                continue
            if self.db.known_review_batch(p):
                candidates.append(p)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _latest_inbox_review_csv(self) -> Path | None:
        inbox_root = (self.root_dir / "ml" / "data" / "inbox").resolve()
        if not inbox_root.exists():
            return None
        candidates = []
        for name in ("false_positive_review.csv", "false_positive_review_labeled.csv"):
            candidates.extend(p for p in inbox_root.rglob(name) if p.is_file())
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _infer_batch_from_csv(self, path: Path) -> Path | None:
        inbox_root = (self.root_dir / "ml" / "data" / "inbox").resolve()
        p = path.resolve()
        for parent in [p] + list(p.parents):
            if parent.parent == inbox_root:
                return parent
        return None

    def _load_rows(self, csv_path: Path) -> list[dict]:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        if not rows:
            if "source_path" in fieldnames:
                return rows
            if "path" in fieldnames:
                return rows
            raise RuntimeError("CSV must contain source_path (or path) column")

        if "source_path" not in rows[0]:
            if "path" in rows[0]:
                for row in rows:
                    row["source_path"] = row["path"]
            else:
                raise RuntimeError("CSV must contain source_path (or path) column")
        return rows

    def _seed_labels_from_csv(self):
        for row in self.rows:
            src = str(row.get("source_path", "")).strip()
            existing = str(row.get("human_label", "")).strip().lower()
            if src and existing in VALID_LABELS and existing:
                self.labels[src] = existing
                if self.context:
                    batch = str(self.active_batch) if self.active_batch else ""
                    self.db.set_label(str(self.context.batch_path), src, existing, batch)

    def load_batch(self, batch_path: Path):
        batch_path = batch_path.resolve()
        rows = self.db.load_review_items(batch_path)
        if not rows and not self.db.known_review_batch(batch_path):
            raise FileNotFoundError(f"No review queue found in DB for batch: {batch_path}")
        self.active_batch = batch_path
        self.context = infer_context_for_batch(batch_path)
        self.rows = rows
        self.auto_labels = self.db.load_auto_clip_labels(batch_path)
        self.labels = dict(self.auto_labels)
        self.labels.update(self.db.load_labels(str(batch_path)))

    def load_csv(self, csv_path: Path):
        csv_path = csv_path.resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        inferred_batch = self._infer_batch_from_csv(csv_path)
        if not inferred_batch:
            raise RuntimeError(f"Legacy review CSV must live under ml/data/inbox/<batch>: {csv_path}")
        self.active_batch = inferred_batch
        self.context = infer_context_for_batch(inferred_batch)
        self.rows = self._load_rows(csv_path)
        self.db.replace_review_batch(inferred_batch, self.rows, review_csv_path=csv_path)
        self.auto_labels = self.db.load_auto_clip_labels(inferred_batch)
        self.labels = dict(self.auto_labels)
        self.labels.update(self.db.load_labels(str(self.context.batch_path)))
        self._seed_labels_from_csv()

    def save_labels(self):
        return

    def set_label(self, source_path: str, label: str):
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label: {label}")
        if label == "":
            if source_path in self.auto_labels:
                self.labels[source_path] = self.auto_labels[source_path]
            else:
                self.labels.pop(source_path, None)
        else:
            self.labels[source_path] = label
        if self.context:
            batch = str(self.active_batch) if self.active_batch else ""
            self.db.set_label(str(self.context.batch_path), source_path, label, batch)
        self.save_labels()

    def status(self):
        total = len(self.rows)
        labeled = 0
        counts = {"piano": 0, "non_piano": 0, "unclear": 0}
        for row in self.rows:
            lbl = self.labels.get(str(row.get("source_path", "")), "")
            if lbl:
                labeled += 1
                if lbl in counts:
                    counts[lbl] += 1
        return {"total": total, "labeled": labeled, "unlabeled": total - labeled, "counts": counts}

    def export_csv(self) -> Path:
        if not self.context:
            raise RuntimeError("No active review batch loaded")
        out_rows = []
        for row in self.rows:
            merged = dict(row)
            merged["human_label"] = self.labels.get(str(row.get("source_path", "")), "")
            out_rows.append(merged)

        fieldnames = list(out_rows[0].keys()) if out_rows else []
        if "human_label" not in fieldnames:
            fieldnames.append("human_label")
        self.context.export_out.parent.mkdir(parents=True, exist_ok=True)
        with self.context.export_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)
        return self.context.export_out

    def items(self):
        out = []
        for idx, row in enumerate(self.rows):
            src = str(row.get("source_path", "")).strip()
            review_score = row.get("review_score", row.get("confidence_from_boundary", ""))
            score_label = "Review Score" if str(review_score).strip() else "Model Piano Prob"
            score_value = review_score if str(review_score).strip() else row.get("piano_probability", "")
            out.append(
                {
                    "index": idx,
                    "source_path": src,
                    "review_rank": row.get("review_rank", ""),
                    "piano_probability": row.get("piano_probability", ""),
                    "score_label": score_label,
                    "score_value": score_value,
                    "predicted_label": row.get("predicted_label", ""),
                    "audio_url": f"/api/audio?path={quote(src, safe='')}",
                    "human_label": self.labels.get(src, ""),
                }
            )
        return out

    def context_payload(self):
        if not self.context:
            return {
                "active_batch": str(self.active_batch) if self.active_batch else "",
                "review_csv_path": "",
                "export_csv": "",
            }
        return {
            "review_csv_path": str(self.context.review_csv_path),
            "export_csv": str(self.context.export_out),
            "active_batch": str(self.active_batch) if self.active_batch else "",
        }


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def create_handler(state: ReviewState, static_dir: Path, db: ActivityDb):
    iterate_script = static_dir / "ml" / "scripts" / "iterate_training.py"
    hard_positive_script = static_dir / "ml" / "scripts" / "build_hard_positive_queue.py"
    inbox_root = (static_dir / "ml" / "data" / "inbox").resolve()
    ops_dir = (static_dir / "ml" / "data" / "ops").resolve()
    review_queue_root = (static_dir / "ml" / "data" / "review" / "queues").resolve()
    inbox_root.mkdir(parents=True, exist_ok=True)
    ops_dir.mkdir(parents=True, exist_ok=True)
    review_queue_root.mkdir(parents=True, exist_ok=True)
    finalize_lock = threading.Lock()
    finalize_proc_slot: dict[str, subprocess.Popen | None] = {"proc": None}
    finalize_cancel_event = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.path = "/ml/tools/clip_reviewer/index.html"
                return super().do_GET()
            if parsed.path == "/api/items":
                return json_response(
                    self,
                    {
                        "items": state.items(),
                        "status": state.status(),
                        "context": state.context_payload(),
                    },
                )
            if parsed.path == "/api/status":
                return json_response(self, {"status": state.status(), "context": state.context_payload()})
            if parsed.path == "/api/audio":
                return self._serve_audio(parsed)
            if parsed.path == "/api/storage/upload_history":
                return self._handle_upload_history(parsed)
            if parsed.path == "/api/storage/analyze":
                return self._handle_storage_analyze(parsed)
            if parsed.path == "/api/long_audio/queue":
                return self._handle_long_audio_queue(parsed)
            return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/batch/new":
                return self._handle_new_batch()
            if parsed.path == "/api/upload_batch":
                return self._handle_upload_batch()
            if parsed.path == "/api/label":
                return self._handle_label()
            if parsed.path == "/api/export":
                return self._handle_export()
            if parsed.path == "/api/load_batch":
                return self._handle_load_batch()
            if parsed.path == "/api/review/build_hard_positive_queue":
                return self._handle_build_hard_positive_queue()
            if parsed.path == "/api/review/load_false_negative_queue":
                return self._handle_load_false_negative_queue()
            if parsed.path == "/api/load_csv":
                return self._handle_load_csv()
            if parsed.path == "/api/pipeline/run_batch":
                return self._handle_run_batch()
            if parsed.path == "/api/pipeline/promote":
                return self._handle_promote()
            if parsed.path == "/api/pipeline/retrain":
                return self._handle_retrain()
            if parsed.path == "/api/pipeline/eval_holdout":
                return self._handle_eval_holdout()
            if parsed.path == "/api/pipeline/audit_dataset":
                return self._handle_audit_dataset()
            if parsed.path == "/api/pipeline/finalize":
                return self._handle_finalize()
            if parsed.path == "/api/pipeline/finalize/cancel":
                return self._handle_finalize_cancel()
            if parsed.path == "/api/storage/retention_cleanup":
                return self._handle_retention_cleanup()
            if parsed.path == "/api/storage/smart_cleanup":
                return self._handle_smart_cleanup()
            if parsed.path == "/api/long_audio/load":
                return self._handle_long_audio_load()
            if parsed.path == "/api/long_audio/save_annotations":
                return self._handle_long_audio_save_annotations()
            if parsed.path == "/api/long_audio/estimate":
                return self._handle_long_audio_estimate()
            if parsed.path == "/api/long_audio/evaluate":
                return self._handle_long_audio_evaluate()
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(body.decode("utf-8"))

        def _resolve_batch_path(self, batch_input: str) -> Path:
            if not batch_input:
                raise ValueError("batch is required")
            batch_path = Path(batch_input)
            if not batch_path.is_absolute():
                batch_path = (static_dir / batch_input).resolve()
            if inbox_root not in batch_path.parents and batch_path != inbox_root:
                raise ValueError(f"batch must be inside {inbox_root}")
            return batch_path

        def _resolve_repo_path(self, raw_input: str, *, must_exist: bool = False) -> Path:
            if not raw_input:
                raise ValueError("path is required")
            p = Path(raw_input)
            if not p.is_absolute():
                p = (static_dir / raw_input).resolve()
            else:
                p = p.resolve()
            if static_dir not in p.parents and p != static_dir:
                raise ValueError(f"path must be inside {static_dir}")
            if must_exist and not p.exists():
                raise FileNotFoundError(f"path not found: {p}")
            return p

        def _resolve_eval_csv_path(self, raw_input: str, *, default_rel: str) -> Path:
            raw = str(raw_input or "").strip() or default_rel
            p = Path(raw)
            if not p.is_absolute():
                p = (static_dir / raw).resolve()
            else:
                p = p.resolve()
            eval_root = (static_dir / "ml" / "data" / "eval").resolve()
            eval_root.mkdir(parents=True, exist_ok=True)
            if eval_root not in p.parents and p != eval_root:
                raise ValueError(f"annotation file must live inside {eval_root}")
            return p

        def _normalize_path_key(self, raw_path: str | Path) -> str:
            try:
                p = Path(raw_path)
                if not p.is_absolute():
                    p = (static_dir / p).resolve()
                else:
                    p = p.resolve()
                return str(p).replace("/", "\\").lower()
            except Exception:
                return str(raw_path).replace("/", "\\").lower()

        def _load_long_audio_queue_rows(self, queue_csv: Path) -> list[dict]:
            if not queue_csv.exists():
                return []
            with queue_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            out = []
            for row in rows:
                audio_path = str(row.get("audio_path", "")).strip()
                if not audio_path:
                    continue
                resolved_audio = self._resolve_repo_path(audio_path, must_exist=False)
                out.append(
                    {
                        **{k: str(v or "") for k, v in row.items()},
                        "audio_path": str(resolved_audio),
                        "audio_exists": resolved_audio.exists(),
                    }
                )
            return out

        def _load_annotation_rows(self, annotations_csv: Path, audio_path: Path) -> list[dict]:
            if not annotations_csv.exists():
                return []
            target_key = self._normalize_path_key(audio_path)
            with annotations_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            out = []
            for row in rows:
                if self._normalize_path_key(str(row.get("audio_path", "")).strip()) != target_key:
                    continue
                out.append(
                    {
                        "audio_path": str(audio_path),
                        "start_seconds": str(row.get("start_seconds", "")).strip(),
                        "end_seconds": str(row.get("end_seconds", "")).strip(),
                        "label": str(row.get("label", "")).strip(),
                        "notes": str(row.get("notes", "")).strip(),
                    }
                )
            return out

        def _save_annotation_rows(self, annotations_csv: Path, audio_path: Path, rows: list[dict]) -> None:
            annotations_csv.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if annotations_csv.exists():
                with annotations_csv.open("r", encoding="utf-8", newline="") as f:
                    existing = list(csv.DictReader(f))

            target_key = self._normalize_path_key(audio_path)
            preserved = [row for row in existing if self._normalize_path_key(str(row.get("audio_path", "")).strip()) != target_key]

            normalized_rows = []
            for row in rows:
                label = str(row.get("label", "")).strip().lower()
                if label not in LONG_AUDIO_LABELS:
                    raise ValueError(f"Unsupported label: {label}")
                start_seconds = str(row.get("start_seconds", "")).strip()
                end_seconds = str(row.get("end_seconds", "")).strip()
                if not start_seconds or not end_seconds:
                    continue
                start_val = float(start_seconds)
                end_val = float(end_seconds)
                if end_val <= start_val:
                    raise ValueError(f"end_seconds must be > start_seconds ({start_val}, {end_val})")
                normalized_rows.append(
                    {
                        "audio_path": str(audio_path.resolve()),
                        "start_seconds": f"{start_val:.3f}",
                        "end_seconds": f"{end_val:.3f}",
                        "label": label,
                        "notes": str(row.get("notes", "")).strip(),
                    }
                )

            all_rows = preserved + normalized_rows
            fieldnames = ["audio_path", "start_seconds", "end_seconds", "label", "notes"]
            with annotations_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)

        def _run_script_and_load_json(
            self,
            cmd: list[str],
            *,
            summary_json_path: Path,
            intervals_csv_path: Path | None = None,
            frames_csv_path: Path | None = None,
        ) -> dict:
            proc = subprocess.run(
                cmd,
                cwd=str(static_dir),
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
            if not summary_json_path.exists():
                raise RuntimeError(f"Expected summary JSON not found: {summary_json_path}")
            summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
            intervals = []
            if intervals_csv_path and intervals_csv_path.exists():
                with intervals_csv_path.open("r", encoding="utf-8", newline="") as f:
                    intervals = list(csv.DictReader(f))
            frames = []
            if frames_csv_path and frames_csv_path.exists():
                with frames_csv_path.open("r", encoding="utf-8", newline="") as f:
                    frames = list(csv.DictReader(f))
            return {
                "summary": summary,
                "intervals": intervals,
                "frames": frames,
                "command_output": (proc.stdout or "").strip(),
            }

        def _create_new_batch_path(self) -> Path:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = inbox_root / f"unseen_batch_{stamp}"
            if not base.exists():
                return base
            i = 1
            while True:
                candidate = inbox_root / f"unseen_batch_{stamp}_{i}"
                if not candidate.exists():
                    return candidate
                i += 1

        def _infer_batch_from_path(self, path: Path) -> Path | None:
            p = path.resolve()
            for parent in [p] + list(p.parents):
                if parent.parent == inbox_root:
                    return parent
            return None

        def _choose_unique_target(self, raw_long_dir: Path, filename: str) -> Path:
            target = raw_long_dir / filename
            if not target.exists():
                return target
            stem = target.stem
            ext = target.suffix
            i = 1
            while True:
                candidate = raw_long_dir / f"{stem}_{i}{ext}"
                if not candidate.exists():
                    return candidate
                i += 1

        def _handle_upload_history(self, parsed):
            try:
                query = parse_qs(parsed.query)
                min_size_mb = float((query.get("min_size_mb") or ["0"])[0])
                limit = int((query.get("limit") or ["200"])[0])
                statuses_raw = str((query.get("statuses") or ["success,duplicate"])[0]).strip()
                statuses = [s.strip() for s in statuses_raw.split(",") if s.strip()]
                rows = db.get_upload_history(
                    min_size_bytes=max(0, int(min_size_mb * 1024 * 1024)),
                    limit=max(1, min(limit, 5000)),
                    statuses=statuses,
                )
                return json_response(
                    self,
                    {
                        "ok": True,
                        "db_path": str(db.db_path),
                        "count": len(rows),
                        "items": rows,
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _dir_summary(self, root: Path) -> dict:
            size_bytes = 0
            file_count = 0
            if root.exists() and root.is_dir():
                for p in root.rglob("*"):
                    if p.is_file():
                        file_count += 1
                        try:
                            size_bytes += p.stat().st_size
                        except OSError:
                            pass
            return {
                "path": str(root),
                "exists": root.exists(),
                "file_count": file_count,
                "size_bytes": int(size_bytes),
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            }

        def _handle_storage_analyze(self, parsed):
            try:
                query = parse_qs(parsed.query)
                top_n = max(5, min(200, int((query.get("top_n") or ["20"])[0])))
                ml_dir = (static_dir / "ml").resolve()
                data_dir = (ml_dir / "data").resolve()
                inbox_dir = (data_dir / "inbox").resolve()

                sections = {}
                for name in ("raw", "inbox", "review", "labels", "processed", "ops"):
                    sections[name] = self._dir_summary(data_dir / name)

                top_files = []
                for p in ml_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    try:
                        sz = p.stat().st_size
                    except OSError:
                        continue
                    top_files.append({"path": str(p), "size_bytes": int(sz)})
                top_files.sort(key=lambda x: x["size_bytes"], reverse=True)
                top_files = [
                    {**x, "size_mb": round(x["size_bytes"] / (1024 * 1024), 2)}
                    for x in top_files[:top_n]
                ]

                batches = []
                if inbox_dir.exists():
                    for b in sorted(inbox_dir.iterdir()):
                        if not b.is_dir():
                            continue
                        raw = self._dir_summary(b / "raw_long")
                        clips = self._dir_summary(b / "clips")
                        batches.append(
                            {
                                "batch": str(b),
                                "raw_long_mb": raw["size_mb"],
                                "clips_mb": clips["size_mb"],
                                "raw_long_files": raw["file_count"],
                                "clips_files": clips["file_count"],
                            }
                        )

                return json_response(
                    self,
                    {
                        "ok": True,
                        "generated_at": utc_now_iso(),
                        "db_overview": db.overview(),
                        "sections": sections,
                        "top_files": top_files,
                        "inbox_batches": batches,
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_long_audio_queue(self, parsed):
            try:
                query = parse_qs(parsed.query)
                queue_csv = self._resolve_eval_csv_path(
                    str((query.get("queue_csv") or ["ml/data/eval/long_audio_annotation_queue.csv"])[0]).strip(),
                    default_rel="ml/data/eval/long_audio_annotation_queue.csv",
                )
                rows = self._load_long_audio_queue_rows(queue_csv)
                return json_response(self, {"ok": True, "queue_csv": str(queue_csv), "items": rows})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_long_audio_load(self):
            try:
                payload = self._read_json()
                audio_path = self._resolve_repo_path(str(payload.get("audio_path", "")).strip(), must_exist=True)
                annotations_csv = self._resolve_eval_csv_path(
                    str(payload.get("annotations_csv", "")).strip(),
                    default_rel="ml/data/eval/long_audio_annotations.csv",
                )
                rows = self._load_annotation_rows(annotations_csv, audio_path)
                return json_response(
                    self,
                    {
                        "ok": True,
                        "audio_path": str(audio_path),
                        "audio_url": f"/api/audio?path={quote(str(audio_path))}",
                        "annotations_csv": str(annotations_csv),
                        "annotation_rows": rows,
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_long_audio_save_annotations(self):
            try:
                payload = self._read_json()
                audio_path = self._resolve_repo_path(str(payload.get("audio_path", "")).strip(), must_exist=True)
                annotations_csv = self._resolve_eval_csv_path(
                    str(payload.get("annotations_csv", "")).strip(),
                    default_rel="ml/data/eval/long_audio_annotations.csv",
                )
                rows = list(payload.get("rows", []) or [])
                self._save_annotation_rows(annotations_csv, audio_path, rows)
                saved_rows = self._load_annotation_rows(annotations_csv, audio_path)
                return json_response(
                    self,
                    {
                        "ok": True,
                        "audio_path": str(audio_path),
                        "annotations_csv": str(annotations_csv),
                        "saved_count": len(saved_rows),
                        "annotation_rows": saved_rows,
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_long_audio_estimate(self):
            try:
                payload = self._read_json()
                audio_path = self._resolve_repo_path(str(payload.get("audio_path", "")).strip(), must_exist=True)
                model = str(payload.get("model", "ml/models/piano_detector_v23_ensemble.joblib")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                stamp = hashlib.sha1(f"{audio_path}|{time.time()}".encode("utf-8")).hexdigest()[:12]
                frames_csv = ops_dir / f"timeline_preview_{stamp}_frames.csv"
                intervals_csv = ops_dir / f"timeline_preview_{stamp}_intervals.csv"
                summary_json = ops_dir / f"timeline_preview_{stamp}_summary.json"
                cmd = [
                    sys.executable,
                    str(static_dir / "ml" / "scripts" / "estimate_piano_timeline.py"),
                    str(audio_path),
                    "--model",
                    model,
                    "--config",
                    config,
                    "--out-frames-csv",
                    str(frames_csv),
                    "--out-intervals-csv",
                    str(intervals_csv),
                    "--out-summary-json",
                    str(summary_json),
                ]
                if str(payload.get("window_seconds", "")).strip():
                    cmd.extend(["--window-seconds", str(payload.get("window_seconds"))])
                if str(payload.get("hop_seconds", "")).strip():
                    cmd.extend(["--hop-seconds", str(payload.get("hop_seconds"))])
                if str(payload.get("smooth_frames", "")).strip():
                    cmd.extend(["--smooth-frames", str(payload.get("smooth_frames"))])
                if payload.get("enter_threshold") is not None and str(payload.get("enter_threshold")).strip() != "":
                    cmd.extend(["--enter-threshold", str(payload.get("enter_threshold"))])
                if payload.get("exit_threshold") is not None and str(payload.get("exit_threshold")).strip() != "":
                    cmd.extend(["--exit-threshold", str(payload.get("exit_threshold"))])

                result = self._run_script_and_load_json(
                    cmd,
                    summary_json_path=summary_json,
                    intervals_csv_path=intervals_csv,
                    frames_csv_path=frames_csv,
                )
                return json_response(
                    self,
                    {
                        "ok": True,
                        "audio_path": str(audio_path),
                        "summary": result["summary"],
                        "intervals": result["intervals"],
                        "frames": result["frames"],
                        "command_output": result["command_output"],
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_long_audio_evaluate(self):
            try:
                payload = self._read_json()
                audio_path = self._resolve_repo_path(str(payload.get("audio_path", "")).strip(), must_exist=True)
                annotations_csv = self._resolve_eval_csv_path(
                    str(payload.get("annotations_csv", "")).strip(),
                    default_rel="ml/data/eval/long_audio_annotations.csv",
                )
                model = str(payload.get("model", "ml/models/piano_detector_v23_ensemble.joblib")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                stamp = hashlib.sha1(f"eval|{audio_path}|{time.time()}".encode("utf-8")).hexdigest()[:12]
                frames_csv = ops_dir / f"timeline_eval_{stamp}_frames.csv"
                intervals_csv = ops_dir / f"timeline_eval_{stamp}_intervals.csv"
                gt_csv = ops_dir / f"timeline_eval_{stamp}_ground_truth.csv"
                summary_json = ops_dir / f"timeline_eval_{stamp}_summary.json"
                cmd = [
                    sys.executable,
                    str(static_dir / "ml" / "scripts" / "evaluate_piano_timeline.py"),
                    str(audio_path),
                    "--annotations-csv",
                    str(annotations_csv),
                    "--model",
                    model,
                    "--config",
                    config,
                    "--out-frames-csv",
                    str(frames_csv),
                    "--out-intervals-csv",
                    str(intervals_csv),
                    "--out-ground-truth-csv",
                    str(gt_csv),
                    "--out-summary-json",
                    str(summary_json),
                ]
                if str(payload.get("window_seconds", "")).strip():
                    cmd.extend(["--window-seconds", str(payload.get("window_seconds"))])
                if str(payload.get("hop_seconds", "")).strip():
                    cmd.extend(["--hop-seconds", str(payload.get("hop_seconds"))])
                if str(payload.get("smooth_frames", "")).strip():
                    cmd.extend(["--smooth-frames", str(payload.get("smooth_frames"))])
                if payload.get("enter_threshold") is not None and str(payload.get("enter_threshold")).strip() != "":
                    cmd.extend(["--enter-threshold", str(payload.get("enter_threshold"))])
                if payload.get("exit_threshold") is not None and str(payload.get("exit_threshold")).strip() != "":
                    cmd.extend(["--exit-threshold", str(payload.get("exit_threshold"))])

                result = self._run_script_and_load_json(
                    cmd,
                    summary_json_path=summary_json,
                    intervals_csv_path=intervals_csv,
                    frames_csv_path=frames_csv,
                )
                return json_response(
                    self,
                    {
                        "ok": True,
                        "audio_path": str(audio_path),
                        "annotations_csv": str(annotations_csv),
                        "summary": result["summary"],
                        "intervals": result["intervals"],
                        "frames": result["frames"],
                        "command_output": result["command_output"],
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_new_batch(self):
            try:
                batch_path = self._create_new_batch_path()
                (batch_path / "raw_long").mkdir(parents=True, exist_ok=True)
                state.active_batch = batch_path
                return json_response(
                    self,
                    {
                        "ok": True,
                        "batch": str(batch_path),
                        "raw_long_dir": str(batch_path / "raw_long"),
                        "context": state.context_payload(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_upload_batch(self):
            try:
                ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
                if ctype != "multipart/form-data":
                    raise ValueError("Content-Type must be multipart/form-data")

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                batch_input = str(form.getfirst("batch", "")).strip()
                upload_label_mode = str(form.getfirst("upload_label_mode", "")).strip().lower()
                if upload_label_mode not in {"", "piano", "non_piano"}:
                    raise ValueError("upload_label_mode must be piano, non_piano, or empty")
                if batch_input:
                    batch_path = self._resolve_batch_path(batch_input)
                elif state.active_batch:
                    batch_path = state.active_batch
                else:
                    batch_path = self._create_new_batch_path()
                raw_long_dir = batch_path / "raw_long"
                raw_long_dir.mkdir(parents=True, exist_ok=True)

                if "files" not in form:
                    raise ValueError("No files uploaded")
                file_fields = form["files"]
                if not isinstance(file_fields, list):
                    file_fields = [file_fields]

                saved = []
                skipped = []
                for field in file_fields:
                    filename = Path(getattr(field, "filename", "")).name
                    if not filename:
                        continue
                    suffix = Path(filename).suffix.lower()
                    if suffix not in AUDIO_EXTS:
                        skipped.append({"file": filename, "reason": "unsupported extension"})
                        db.log_upload(
                            file_id=None,
                            batch_path=batch_path,
                            source_name=filename,
                            stored_path="",
                            status=STATUS_SKIPPED,
                            reason="unsupported extension",
                        )
                        continue

                    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".upload_", suffix=suffix, dir=str(raw_long_dir))
                    os.close(tmp_fd)
                    Path(tmp_name).unlink(missing_ok=True)
                    tmp_path = Path(tmp_name)
                    try:
                        size_bytes, sha256 = copy_stream_to_file_with_hash(field.file, tmp_path)
                        existing = db.find_success_by_sha(sha256)
                        existing_path = str((existing or {}).get("stored_path", "")).strip()
                        if existing_path and Path(existing_path).exists():
                            tmp_path.unlink(missing_ok=True)
                            file_id = db.upsert_file(sha256=sha256, size_bytes=size_bytes, original_name=filename)
                            db.log_upload(
                                file_id=file_id,
                                batch_path=batch_path,
                                source_name=filename,
                                stored_path=existing_path,
                                status=STATUS_DUPLICATE,
                                reason="duplicate content hash",
                            )
                            skipped.append(
                                {
                                    "file": filename,
                                    "reason": "duplicate content hash",
                                    "existing_path": existing_path,
                                }
                            )
                            continue

                        target = self._choose_unique_target(raw_long_dir, filename)
                        shutil.move(str(tmp_path), str(target))
                        file_id = db.upsert_file(sha256=sha256, size_bytes=size_bytes, original_name=filename)
                        db.log_upload(
                            file_id=file_id,
                            batch_path=batch_path,
                            source_name=filename,
                            stored_path=str(target),
                            status=STATUS_SUCCESS,
                        )
                        if upload_label_mode:
                            db.set_upload_file_label(batch_path, target, upload_label_mode)
                        saved.append(str(target))
                    except Exception as file_ex:
                        tmp_path.unlink(missing_ok=True)
                        db.log_upload(
                            file_id=None,
                            batch_path=batch_path,
                            source_name=filename,
                            stored_path="",
                            status=STATUS_FAILED,
                            reason=str(file_ex),
                        )
                        skipped.append({"file": filename, "reason": f"upload failed: {file_ex}"})

                if not saved:
                    raise RuntimeError("No valid audio files were uploaded")
                state.active_batch = batch_path

                return json_response(
                    self,
                    {
                        "ok": True,
                        "batch": str(batch_path),
                        "raw_long_dir": str(raw_long_dir),
                        "saved_count": len(saved),
                        "saved_files": saved,
                        "upload_label_mode": upload_label_mode,
                        "skipped": skipped,
                        "db_path": str(db.db_path),
                        "context": state.context_payload(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _serve_audio(self, parsed):
            query = parse_qs(parsed.query)
            raw_path = (query.get("path") or [""])[0]
            if not raw_path:
                return json_response(self, {"ok": False, "error": "missing path"}, status=400)

            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = (static_dir / raw_path).resolve()
            if not audio_path.exists() or not audio_path.is_file():
                return json_response(self, {"ok": False, "error": f"audio not found: {audio_path}"}, status=404)
            if audio_path.suffix.lower() not in AUDIO_EXTS:
                return json_response(self, {"ok": False, "error": "unsupported audio extension"}, status=400)

            data = audio_path.read_bytes()
            mime, _ = mimetypes.guess_type(str(audio_path))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle_load_batch(self):
            try:
                payload = self._read_json()
                batch_input = str(payload.get("batch", "")).strip()
                batch_path = self._resolve_batch_path(batch_input)
                state.load_batch(batch_path)
                return json_response(
                    self,
                    {"ok": True, "status": state.status(), "context": state.context_payload(), "items": state.items()},
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_load_csv(self):
            try:
                payload = self._read_json()
                csv_path = Path(str(payload.get("csv_path", "")).strip())
                if not csv_path:
                    raise ValueError("csv_path is required")
                if not csv_path.is_absolute():
                    csv_path = (static_dir / csv_path).resolve()
                state.load_csv(csv_path)
                inferred_batch = self._infer_batch_from_path(csv_path)
                if inferred_batch:
                    state.active_batch = inferred_batch
                return json_response(
                    self,
                    {"ok": True, "status": state.status(), "context": state.context_payload(), "items": state.items()},
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _load_queue_csv_into_state(self, queue_name: str, queue_csv: Path):
            rows = state._load_rows(queue_csv)
            queue_batch = (review_queue_root / queue_name).resolve()
            queue_batch.mkdir(parents=True, exist_ok=True)
            db.replace_review_batch(queue_batch, rows, review_csv_path=queue_csv)
            state.load_batch(queue_batch)

        def _handle_load_false_negative_queue(self):
            try:
                payload = self._read_json()
                queue_name = str(payload.get("queue_name", "holdout_false_negatives")).strip() or "holdout_false_negatives"
                source_csv = Path(
                    str(payload.get("source_csv", "ml/reports/piano_detector_v23_ensemble_false_negatives.csv")).strip()
                )
                threshold = float(payload.get("threshold", 0.84))
                if not source_csv.is_absolute():
                    source_csv = (static_dir / source_csv).resolve()
                if not source_csv.exists():
                    raise FileNotFoundError(f"False-negative CSV not found: {source_csv}")

                queue_dir = (review_queue_root / queue_name).resolve()
                queue_dir.mkdir(parents=True, exist_ok=True)
                queue_csv = queue_dir / "queue.csv"

                rows = []
                with source_csv.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for rank, row in enumerate(reader, start=1):
                        src = str(row.get("resolved_source_path", "")).strip()
                        if not src:
                            continue
                        try:
                            prob = float(row.get("ensemble_probability", row.get("piano_probability", "")))
                        except Exception:
                            prob = None
                        rows.append(
                            {
                                "source_path": src,
                                "review_rank": rank,
                                "review_score": abs(prob - threshold) if prob is not None else "",
                                "piano_probability": prob if prob is not None else "",
                                "predicted_label": "non_piano",
                            }
                        )
                if not rows:
                    raise RuntimeError("False-negative CSV contains no playable source paths")

                with queue_csv.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["source_path", "review_rank", "review_score", "piano_probability", "predicted_label"],
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

                self._load_queue_csv_into_state(queue_name, queue_csv)
                return json_response(
                    self,
                    {
                        "ok": True,
                        "queue_csv": str(queue_csv),
                        "status": state.status(),
                        "context": state.context_payload(),
                        "items": state.items(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_build_hard_positive_queue(self):
            try:
                payload = self._read_json()
                top_k = str(payload.get("top_k", 120))
                max_prob = str(payload.get("max_prob", 0.75))
                model = str(payload.get("model", "ml/models/piano_detector_v23_ensemble.joblib")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                queue_name = "hard_positive_mainline"
                queue_dir = (review_queue_root / queue_name).resolve()
                queue_dir.mkdir(parents=True, exist_ok=True)
                queue_csv = queue_dir / "queue.csv"

                cmd = [
                    sys.executable,
                    str(hard_positive_script),
                    "--asset-index",
                    "ml/data/review/asset_pool/train_index.csv",
                    "--model",
                    model,
                    "--config",
                    config,
                    "--out-csv",
                    str(queue_csv),
                    "--top-k",
                    top_k,
                    "--max-prob",
                    max_prob,
                ]
                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "build_hard_positive_queue failed")
                self._load_queue_csv_into_state(queue_name, queue_csv)
                return json_response(
                    self,
                    {
                        "ok": True,
                        "command_output": result["stdout"],
                        "queue_csv": str(queue_csv),
                        "status": state.status(),
                        "context": state.context_payload(),
                        "items": state.items(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_label(self):
            try:
                payload = self._read_json()
                source_path = str(payload.get("source_path", "")).strip()
                label = str(payload.get("label", "")).strip().lower()
                if not source_path:
                    raise ValueError("source_path is required")
                state.set_label(source_path, label)
                return json_response(self, {"ok": True, "status": state.status(), "context": state.context_payload()})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_export(self):
            try:
                out_csv = state.export_csv()
                return json_response(
                    self,
                    {
                        "ok": True,
                        "export_csv": str(out_csv),
                        "status": state.status(),
                        "context": state.context_payload(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_run_batch(self):
            try:
                payload = self._read_json()
                batch = str(payload.get("batch", "")).strip()
                if batch:
                    batch_path = self._resolve_batch_path(batch)
                    state.active_batch = batch_path
                elif state.active_batch:
                    batch_path = state.active_batch
                else:
                    raise ValueError("No active batch. Upload files first.")

                model = str(payload.get("model", "ml/models/piano_detector_v23_ensemble.joblib")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                false_positive_threshold_raw = payload.get("false_positive_threshold", None)
                top_k = str(payload.get("top_k", 120))
                clip_seconds = str(payload.get("clip_seconds", 8))
                hop_seconds = str(payload.get("hop_seconds", 8))
                min_rms_db = str(payload.get("min_rms_db", -50))
                label_prefix = str(payload.get("label_prefix", "")).strip()
                bucket = str(payload.get("bucket", "")).strip()
                if not bucket:
                    bucket = batch_path.name
                auto_promote = bool(payload.get("auto_promote_all_non_piano", False))

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "run-batch",
                    "--batch",
                    str(batch_path),
                    "--model",
                    model,
                    "--config",
                    config,
                    "--top-k",
                    top_k,
                    "--clip-seconds",
                    clip_seconds,
                    "--hop-seconds",
                    hop_seconds,
                    "--min-rms-db",
                    min_rms_db,
                ]
                if false_positive_threshold_raw not in {None, ""}:
                    cmd.extend(["--false-positive-threshold", str(false_positive_threshold_raw)])
                if label_prefix:
                    cmd.extend(["--label-prefix", label_prefix])
                if bucket:
                    cmd.extend(["--bucket", bucket])
                if auto_promote:
                    cmd.append("--auto-promote-all-non-piano")

                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "run-batch failed")

                summary_path = batch_path / "iteration_summary.json"
                if not summary_path.exists():
                    raise RuntimeError("iteration_summary.json was not produced")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))

                state.load_batch(batch_path)
                return json_response(
                    self,
                    {
                        "ok": True,
                        "summary": summary,
                        "command_output": result["stdout"],
                        "status": state.status(),
                        "context": state.context_payload(),
                        "items": state.items(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_promote(self):
            try:
                payload = self._read_json()
                bucket = str(payload.get("bucket", "")).strip()
                if not bucket and state.active_batch:
                    bucket = state.active_batch.name
                non_piano_target = str(payload.get("non_piano_target", "ml/data/raw/non_piano/hard_negative")).strip()
                piano_target = str(payload.get("piano_target", "ml/data/raw/piano/hard_positive")).strip()
                if not state.active_batch:
                    raise ValueError("No active batch to promote")

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "promote-labels",
                    "--batch",
                    str(state.active_batch),
                    "--non-piano-target",
                    non_piano_target,
                    "--piano-target",
                    piano_target,
                ]
                if bucket:
                    cmd.extend(["--bucket", bucket])

                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "promote failed")
                return json_response(self, {"ok": True, "command_output": result["stdout"], "context": state.context_payload()})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_retrain(self):
            try:
                payload = self._read_json()
                raw_root = str(payload.get("raw_root", "ml/data/raw")).strip()
                manifest_out = str(payload.get("manifest_out", "ml/data/labels/manifest.csv")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                model_out = str(payload.get("model_out", "ml/models/piano_detector_v2.joblib")).strip()
                report_out = str(payload.get("report_out", "ml/reports/piano_detector_v2_metrics.json")).strip()
                lower_model_out = model_out.lower()
                if "ensemble" in lower_model_out:
                    v2_model = str(payload.get("v2_model", "ml/models/piano_detector_v2.joblib")).strip()
                    v2_config = str(payload.get("v2_config", "ml/configs/train_config_v2.yaml")).strip()
                    v2_report = str(payload.get("v2_report_out", "ml/reports/piano_detector_v2_metrics.json")).strip()
                    v3_model = str(payload.get("v3_model", "ml/models/piano_detector_v3.joblib")).strip()
                    v3_config = str(payload.get("v3_config", "ml/configs/train_config_v3.yaml")).strip()
                    v3_report = str(payload.get("v3_report_out", "ml/reports/piano_detector_v3_metrics.json")).strip()

                    commands = [
                        [
                            sys.executable,
                            str(iterate_script),
                            "retrain-v2",
                            "--raw-root",
                            raw_root,
                            "--manifest-out",
                            manifest_out,
                            "--config",
                            v2_config,
                            "--model-out",
                            v2_model,
                            "--report-out",
                            v2_report,
                        ],
                        [
                            sys.executable,
                            str(iterate_script),
                            "retrain-v3",
                            "--raw-root",
                            raw_root,
                            "--manifest-out",
                            manifest_out,
                            "--config",
                            v3_config,
                            "--model-out",
                            v3_model,
                            "--report-out",
                            v3_report,
                        ],
                        [
                            sys.executable,
                            str(iterate_script),
                            "build-ensemble-v23",
                            "--v2-model",
                            v2_model,
                            "--v2-config",
                            v2_config,
                            "--v3-model",
                            v3_model,
                            "--v3-config",
                            v3_config,
                            "--model-out",
                            model_out,
                            "--report-out",
                            report_out,
                            "--weight-v2",
                            str(payload.get("weight_v2", 0.7)),
                            "--weight-v3",
                            str(payload.get("weight_v3", 0.3)),
                            "--threshold",
                            str(payload.get("threshold", 0.84)),
                            "--review-threshold",
                            str(payload.get("review_threshold", 0.72)),
                            "--method",
                            str(payload.get("ensemble_method", "weighted_average")).strip(),
                        ],
                    ]
                    outputs = []
                    for cmd in commands:
                        result = run_command(cmd, cwd=static_dir)
                        if result["returncode"] != 0:
                            raise RuntimeError(result["stderr"] or result["stdout"] or "retrain failed")
                        if result["stdout"]:
                            outputs.append(result["stdout"].strip())
                    command_output = "\n\n".join(part for part in outputs if part)
                else:
                    retrain_cmd = "retrain-v3" if ("train_config_v3" in config.lower() or "_v3" in lower_model_out) else "retrain-v2"
                    cmd = [
                        sys.executable,
                        str(iterate_script),
                        retrain_cmd,
                        "--raw-root",
                        raw_root,
                        "--manifest-out",
                        manifest_out,
                        "--config",
                        config,
                        "--model-out",
                        model_out,
                        "--report-out",
                        report_out,
                    ]
                    result = run_command(cmd, cwd=static_dir)
                    if result["returncode"] != 0:
                        raise RuntimeError(result["stderr"] or result["stdout"] or "retrain failed")
                    command_output = result["stdout"]
                return json_response(self, {"ok": True, "command_output": command_output, "context": state.context_payload()})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_eval_holdout(self):
            try:
                payload = self._read_json()
                holdout_root = str(payload.get("holdout_root", "ml/data/eval/holdout")).strip()
                model = str(payload.get("model", "ml/models/piano_detector_v23_ensemble.joblib")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                out_report = str(payload.get("out_report", "ml/reports/piano_detector_v23_ensemble_holdout_metrics.json")).strip()
                out_preds = str(payload.get("out_preds", "ml/reports/piano_detector_v23_ensemble_holdout_predictions.csv")).strip()

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "eval-holdout",
                    "--holdout-root",
                    holdout_root,
                    "--model",
                    model,
                    "--config",
                    config,
                    "--out-report",
                    out_report,
                    "--out-preds",
                    out_preds,
                ]
                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "eval-holdout failed")

                report_path = Path(out_report)
                if not report_path.is_absolute():
                    report_path = (static_dir / report_path).resolve()
                report_payload = {}
                if report_path.exists():
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

                return json_response(
                    self,
                    {
                        "ok": True,
                        "command_output": result["stdout"],
                        "report_path": str(report_path),
                        "report": report_payload,
                        "context": state.context_payload(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_audit_dataset(self):
            try:
                payload = self._read_json()
                manifest = str(payload.get("manifest", "ml/data/labels/manifest.csv")).strip()
                out_json = str(payload.get("out_json", "ml/reports/dataset_audit.json")).strip()
                out_suspects = str(payload.get("out_suspects_csv", "ml/reports/dataset_suspects.csv")).strip()
                out_conflicts = str(payload.get("out_conflicts_csv", "ml/reports/dataset_conflicts.csv")).strip()

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "audit-dataset",
                    "--manifest",
                    manifest,
                    "--out-json",
                    out_json,
                    "--out-suspects-csv",
                    out_suspects,
                    "--out-conflicts-csv",
                    out_conflicts,
                ]
                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "audit-dataset failed")

                report_path = Path(out_json)
                if not report_path.is_absolute():
                    report_path = (static_dir / report_path).resolve()
                report_payload = {}
                if report_path.exists():
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

                return json_response(
                    self,
                    {
                        "ok": True,
                        "command_output": result["stdout"],
                        "report_path": str(report_path),
                        "report": report_payload,
                        "suspects_csv": str((static_dir / out_suspects).resolve()) if not Path(out_suspects).is_absolute() else out_suspects,
                        "conflicts_csv": str((static_dir / out_conflicts).resolve()) if not Path(out_conflicts).is_absolute() else out_conflicts,
                        "context": state.context_payload(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_finalize(self):
            try:
                with finalize_lock:
                    running = finalize_proc_slot["proc"] is not None and finalize_proc_slot["proc"].poll() is None
                if running:
                    raise RuntimeError("Finalize is already running")
                finalize_cancel_event.clear()
                payload = self._read_json()
                if not state.active_batch:
                    raise ValueError("No active batch to finalize")

                batch = str(state.active_batch)
                bucket = str(payload.get("bucket", "")).strip()
                if not bucket:
                    bucket = state.active_batch.name

                non_piano_target = str(payload.get("non_piano_target", "ml/data/raw/non_piano/hard_negative")).strip()
                piano_target = str(payload.get("piano_target", "ml/data/raw/piano/hard_positive")).strip()
                raw_root = str(payload.get("raw_root", "ml/data/raw")).strip()
                manifest_out = str(payload.get("manifest_out", "ml/data/labels/manifest.csv")).strip()
                config = str(payload.get("config", "ml/configs/train_config_v2.yaml")).strip()
                model_out = str(payload.get("model_out", "ml/models/piano_detector_v2.joblib")).strip()
                report_out = str(payload.get("report_out", "ml/reports/piano_detector_v2_metrics.json")).strip()

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "finalize-batch",
                    "--batch",
                    batch,
                    "--bucket",
                    bucket,
                    "--non-piano-target",
                    non_piano_target,
                    "--piano-target",
                    piano_target,
                    "--raw-root",
                    raw_root,
                    "--manifest-out",
                    manifest_out,
                    "--config",
                    config,
                    "--model-out",
                    model_out,
                    "--report-out",
                    report_out,
                    "--train-script",
                    "train_v2.py",
                ]
                result = run_command_cancellable(
                    cmd,
                    cwd=static_dir,
                    cancel_event=finalize_cancel_event,
                    proc_lock=finalize_lock,
                    proc_slot=finalize_proc_slot,
                )
                if result.get("cancelled"):
                    raise RuntimeError("Finalize cancelled by user")
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "finalize failed")
                return json_response(self, {"ok": True, "command_output": result["stdout"], "context": state.context_payload()})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_finalize_cancel(self):
            try:
                with finalize_lock:
                    proc = finalize_proc_slot["proc"]
                    running = proc is not None and proc.poll() is None
                if not running:
                    return json_response(self, {"ok": True, "message": "No finalize job is running"})
                finalize_cancel_event.set()
                return json_response(self, {"ok": True, "message": "Finalize cancellation requested"})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_retention_cleanup(self):
            try:
                payload = self._read_json()
                ttl_days = int(payload.get("ttl_days", 14))
                delete_batch_completely = bool(payload.get("delete_batch_completely", False))

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "cleanup-retention",
                    "--inbox-root",
                    "ml/data/inbox",
                    "--ttl-days",
                    str(ttl_days),
                ]
                if delete_batch_completely:
                    cmd.append("--delete-batch-completely")
                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "retention cleanup failed")
                return json_response(self, {"ok": True, "command_output": result["stdout"], "context": state.context_payload()})
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

        def _handle_smart_cleanup(self):
            try:
                payload = self._read_json()
                ttl_days = int(payload.get("ttl_days", 14))
                delete_batch_completely = bool(payload.get("delete_batch_completely", False))
                remove_pycache = bool(payload.get("remove_pycache", True))
                remove_upload_temps = bool(payload.get("remove_upload_temps", True))

                removed_dirs = []
                removed_files = []

                if remove_pycache:
                    for p in (static_dir / "ml").rglob("__pycache__"):
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                            removed_dirs.append(str(p))

                if remove_upload_temps:
                    for p in inbox_root.rglob(".upload_*"):
                        if p.is_file():
                            try:
                                p.unlink()
                                removed_files.append(str(p))
                            except OSError:
                                pass

                cmd = [
                    sys.executable,
                    str(iterate_script),
                    "cleanup-retention",
                    "--inbox-root",
                    "ml/data/inbox",
                    "--ttl-days",
                    str(ttl_days),
                ]
                if delete_batch_completely:
                    cmd.append("--delete-batch-completely")
                result = run_command(cmd, cwd=static_dir)
                if result["returncode"] != 0:
                    raise RuntimeError(result["stderr"] or result["stdout"] or "smart cleanup retention step failed")

                return json_response(
                    self,
                    {
                        "ok": True,
                        "removed_dirs_count": len(removed_dirs),
                        "removed_files_count": len(removed_files),
                        "removed_dirs": removed_dirs,
                        "removed_files": removed_files,
                        "retention_output": result["stdout"],
                        "db_overview": db.overview(),
                    },
                )
            except Exception as ex:
                return json_response(self, {"ok": False, "error": str(ex)}, status=400)

    return Handler


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root not found: {root_dir}")

    initial_csv = Path(args.csv).resolve() if args.csv else None
    if initial_csv and not initial_csv.exists():
        raise FileNotFoundError(f"CSV not found: {initial_csv}")

    db = ActivityDb((root_dir / "ml" / "data" / "ops" / "activity.db").resolve())
    state = ReviewState(root_dir=root_dir, initial_csv=initial_csv, db=db)
    handler = create_handler(state=state, static_dir=root_dir, db=db)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Dashboard running on http://{args.host}:{args.port}")
    print(f"Activity DB: {db.db_path}")
    if initial_csv:
        print(f"Initial review CSV: {initial_csv}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
