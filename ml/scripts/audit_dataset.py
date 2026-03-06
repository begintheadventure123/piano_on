import argparse
import json
from pathlib import Path

import pandas as pd


def load_manifest(manifest: Path) -> pd.DataFrame:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    df = pd.read_csv(manifest)
    need = {"path", "label"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")
    return df


def suspicious_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for row in df.itertuples(index=False):
        p = str(row.path).lower()
        lbl = int(row.label)
        if lbl == 0 and ("\\non_piano\\" in p) and ("piano_home_recording" in p):
            out.append({"path": row.path, "label": lbl, "reason": "non_piano path contains piano_home_recording"})
        if lbl == 1 and ("\\piano\\" in p) and ("non_piano_" in Path(str(row.path)).name.lower()):
            out.append({"path": row.path, "label": lbl, "reason": "piano label but filename contains non_piano"})
    return pd.DataFrame(out)


def cross_label_basename_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["basename"] = tmp["path"].map(lambda p: Path(str(p)).name.lower())
    agg = tmp.groupby("basename")["label"].nunique().reset_index()
    bad = set(agg[agg["label"] > 1]["basename"].tolist())
    if not bad:
        return pd.DataFrame(columns=["basename", "path", "label"])
    return tmp[tmp["basename"].isin(bad)][["basename", "path", "label"]].sort_values(["basename", "label", "path"])


def main():
    parser = argparse.ArgumentParser(description="Audit dataset quality risks for piano detector")
    parser.add_argument("--manifest", default="ml/data/labels/manifest.csv")
    parser.add_argument("--out-json", default="ml/reports/dataset_audit.json")
    parser.add_argument("--out-suspects-csv", default="ml/reports/dataset_suspects.csv")
    parser.add_argument("--out-conflicts-csv", default="ml/reports/dataset_conflicts.csv")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    df = load_manifest(manifest)
    suspects = suspicious_rows(df)
    conflicts = cross_label_basename_conflicts(df)

    by_label = df["label"].value_counts().to_dict()
    by_top_parent = df["path"].map(lambda p: str(Path(str(p)).parent)).value_counts().head(20).to_dict()
    report = {
        "manifest": str(manifest.resolve()),
        "num_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in by_label.items()},
        "top_parent_folders": by_top_parent,
        "num_suspicious_rows": int(len(suspects)),
        "num_cross_label_basename_rows": int(len(conflicts)),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_s = Path(args.out_suspects_csv)
    out_s.parent.mkdir(parents=True, exist_ok=True)
    suspects.to_csv(out_s, index=False)

    out_c = Path(args.out_conflicts_csv)
    out_c.parent.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(out_c, index=False)

    print(f"rows={len(df)}")
    print(f"suspects={len(suspects)}")
    print(f"cross_label_basename_rows={len(conflicts)}")
    print(f"report={out_json}")
    print(f"suspects_csv={out_s}")
    print(f"conflicts_csv={out_c}")


if __name__ == "__main__":
    main()
