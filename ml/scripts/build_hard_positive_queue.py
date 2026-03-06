import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from inference_utils import load_effective_config, load_model_pack, predict_probability


QUEUE_PREFIX = "ml/data/review/queues"


def load_asset_index(asset_index: Path) -> pd.DataFrame:
    if not asset_index.exists():
        raise FileNotFoundError(f"Asset index not found: {asset_index}")
    return pd.read_csv(asset_index)


def collect_known_asset_paths(df: pd.DataFrame) -> set[str]:
    known: set[str] = set()
    for col in ["managed_path", "canonical_source_path", "original_source_path", "promoted_path", "review_copy_path"]:
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            value = value.strip()
            if value and value.lower() != "nan":
                known.add(str(Path(value).resolve()))
    return known


def load_reviewed_source_paths(ops_db: Path, reviewed_queue_key: str) -> set[str]:
    if not reviewed_queue_key or not ops_db.exists():
        return set()
    with sqlite3.connect(ops_db) as conn:
        rows = conn.execute(
            """
            SELECT source_path
            FROM review_labels
            WHERE csv_path = ?
              AND label IN ('piano', 'non_piano', 'unclear')
            """,
            (reviewed_queue_key,),
        ).fetchall()
    return {
        str(row[0]).strip()
        for row in rows
        if row and str(row[0]).strip()
    }


def load_db_review_candidates(ops_db: Path, known_asset_paths: set[str], excluded_reviewed: set[str]) -> pd.DataFrame:
    if not ops_db.exists():
        return pd.DataFrame()
    queue_root = str((Path(QUEUE_PREFIX)).resolve()).lower()
    with sqlite3.connect(ops_db) as conn:
        query = """
        SELECT DISTINCT l.csv_path, l.source_path, l.label, i.review_rank, i.review_score, i.piano_probability
        FROM review_labels l
        LEFT JOIN review_items i
          ON i.batch_path = l.csv_path AND i.source_path = l.source_path
        WHERE l.label = 'piano'
        """
        df = pd.read_sql_query(query, conn)
    if df.empty:
        return df
    df["csv_path"] = df["csv_path"].astype(str).str.strip()
    df["source_path"] = df["source_path"].astype(str).str.strip()
    df = df[df["csv_path"].str.lower().str.startswith(queue_root) == False].copy()
    df["resolved_source_path"] = df["source_path"].map(lambda p: str(Path(p).resolve()) if p else "")
    df = df[df["resolved_source_path"].str.len() > 0].copy()
    df = df[df["resolved_source_path"].map(lambda p: Path(p).exists())].copy()
    df = df[~df["resolved_source_path"].isin(known_asset_paths)].copy()
    df = df[~df["resolved_source_path"].isin(excluded_reviewed)].copy()
    df["managed_path"] = df["resolved_source_path"]
    df["dataset_id"] = df["csv_path"].map(lambda p: Path(p).name if p else "")
    df["source_kind"] = "db_review"
    df["original_source_path"] = df["resolved_source_path"]
    return df[
        ["managed_path", "dataset_id", "source_kind", "original_source_path", "review_rank", "review_score", "piano_probability"]
    ].drop_duplicates(subset=["managed_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hard-positive queue from managed labeled piano assets")
    parser.add_argument("--asset-index", default="ml/data/review/asset_pool/train_index.csv")
    parser.add_argument("--model", default="ml/models/piano_detector_v2.joblib")
    parser.add_argument("--config", default="ml/configs/train_config_v2.yaml")
    parser.add_argument("--out-csv", default="ml/data/review/hard_positive_queue.csv")
    parser.add_argument("--cache-csv", default="ml/data/review/hard_positive_queue_cache.csv")
    parser.add_argument("--ops-db", default="ml/data/ops/activity.db")
    parser.add_argument(
        "--exclude-reviewed-queue",
        default="ml/data/review/queues/hard_positive_v2",
        help="DB review queue key whose already-labeled items should be skipped",
    )
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--max-prob", type=float, default=0.75)
    parser.add_argument(
        "--source-kinds",
        default="source,review_copy",
        help="comma-separated source kinds to prioritize for hard-positive review",
    )
    args = parser.parse_args()

    asset_index = Path(args.asset_index)
    asset_df = load_asset_index(asset_index)
    known_asset_paths = collect_known_asset_paths(asset_df)
    asset_df = asset_df[asset_df["human_label"].astype(str).str.lower() == "piano"].copy()
    wanted_source_kinds = {
        part.strip().lower()
        for part in str(args.source_kinds or "").split(",")
        if part.strip()
    }
    if wanted_source_kinds:
        asset_df = asset_df[asset_df["source_kind"].astype(str).str.lower().isin(wanted_source_kinds)].copy()

    ops_db = Path(args.ops_db)
    reviewed_queue = str(args.exclude_reviewed_queue or "").strip()
    reviewed_queue_key = str(Path(reviewed_queue).resolve()) if reviewed_queue else ""
    reviewed_source_paths = load_reviewed_source_paths(ops_db, reviewed_queue_key)

    db_review_df = load_db_review_candidates(ops_db, known_asset_paths, reviewed_source_paths)
    candidate_source = "db_review"
    df = db_review_df.copy()
    if df.empty:
        candidate_source = "asset_pool"
        df = asset_df.copy()
        if reviewed_source_paths:
            df = df[~df["managed_path"].astype(str).isin(reviewed_source_paths)].copy()
    if df.empty:
        raise RuntimeError("No hard-positive candidates found after excluding reviewed items")

    model_path = Path(args.model).resolve()
    pack = load_model_pack(model_path)
    cfg = load_effective_config(pack, args.config)
    model_key = f"{model_path}|{model_path.stat().st_mtime_ns}"

    cache_csv = Path(args.cache_csv)
    cache_index: dict[tuple[str, str, int], dict] = {}
    if cache_csv.exists():
        try:
            cache_df = pd.read_csv(cache_csv)
            for cache_row in cache_df.to_dict(orient="records"):
                managed_path = str(cache_row.get("managed_path", "")).strip()
                cache_model_key = str(cache_row.get("model_key", "")).strip()
                path_mtime_ns = int(cache_row.get("path_mtime_ns", 0) or 0)
                if managed_path and cache_model_key and path_mtime_ns:
                    cache_index[(managed_path, cache_model_key, path_mtime_ns)] = cache_row
        except Exception:
            cache_index = {}

    rows = []
    cache_rows = []
    cache_hits = 0
    for row in df.itertuples(index=False):
        audio_path = str(getattr(row, "managed_path", "") or "").strip()
        if not audio_path:
            continue
        audio_stat = Path(audio_path).stat()
        cache_key = (audio_path, model_key, int(audio_stat.st_mtime_ns))
        cached = cache_index.get(cache_key)
        if cached:
            cache_hits += 1
            rows.append(
                {
                    "review_rank": 0,
                    "source_path": audio_path,
                    "managed_path": audio_path,
                    "dataset_id": str(getattr(row, "dataset_id", "") or ""),
                    "source_kind": str(getattr(row, "source_kind", "") or ""),
                    "original_source_path": str(getattr(row, "original_source_path", "") or ""),
                    "piano_probability_raw": float(cached["piano_probability_raw"]),
                    "piano_probability": float(cached["piano_probability"]),
                    "predicted_label": str(cached["predicted_label"]),
                }
            )
            cache_rows.append(cached)
            continue
        try:
            raw_prob, prob = predict_probability(audio_path, pack, cfg)
            scored_row = {
                "review_rank": 0,
                "source_path": audio_path,
                "managed_path": audio_path,
                "dataset_id": str(getattr(row, "dataset_id", "") or ""),
                "source_kind": str(getattr(row, "source_kind", "") or ""),
                "original_source_path": str(getattr(row, "original_source_path", "") or ""),
                "piano_probability_raw": raw_prob,
                "piano_probability": prob,
                "predicted_label": "piano" if prob >= 0.5 else "non_piano",
            }
            rows.append(scored_row)
            cache_rows.append(
                {
                    "managed_path": audio_path,
                    "model_key": model_key,
                    "path_mtime_ns": int(audio_stat.st_mtime_ns),
                    "piano_probability_raw": raw_prob,
                    "piano_probability": prob,
                    "predicted_label": scored_row["predicted_label"],
                }
            )
        except Exception as ex:
            rows.append(
                {
                    "review_rank": 0,
                    "source_path": audio_path,
                    "managed_path": audio_path,
                    "dataset_id": str(getattr(row, "dataset_id", "") or ""),
                    "source_kind": str(getattr(row, "source_kind", "") or ""),
                    "original_source_path": str(getattr(row, "original_source_path", "") or ""),
                    "piano_probability_raw": float("nan"),
                    "piano_probability": float("nan"),
                    "predicted_label": "error",
                    "error": str(ex),
                }
            )

    scored = pd.DataFrame(rows)
    scored = scored.dropna(subset=["piano_probability"]).copy()
    queue = scored[scored["piano_probability"] <= float(args.max_prob)].copy()
    queue = queue.sort_values(by=["piano_probability", "piano_probability_raw"], ascending=[True, True]).head(max(0, int(args.top_k)))
    queue["review_rank"] = range(1, len(queue) + 1)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(out_csv, index=False)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache_rows).drop_duplicates(subset=["managed_path", "model_key", "path_mtime_ns"], keep="last").to_csv(cache_csv, index=False)

    print(f"asset_index={asset_index.resolve()}")
    print(f"model={model_path}")
    print(f"candidate_source={candidate_source}")
    print(f"excluded_reviewed={len(reviewed_source_paths)}")
    print(f"candidates_scanned={len(df)}")
    print(f"cache_hits={cache_hits}")
    print(f"queue_size={len(queue)}")
    print(f"out_csv={out_csv.resolve()}")


if __name__ == "__main__":
    main()
