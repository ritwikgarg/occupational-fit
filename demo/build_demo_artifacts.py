from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ------------ paths ------------
ROOT = Path(__file__).resolve().parents[1]   # project root if demo/ is under it
DATASET = ROOT / "data" / "final_job_match_dataset.csv"
OUTDIR = Path(__file__).resolve().parent / "data"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------ load dataset ------------
df = pd.read_csv(DATASET)

# Basic cleanup
for c in ["user_doc", "soc_doc"]:
    if c in df.columns:
        df[c] = df[c].fillna("")
else:
    # It's okay if not present; the demo will still work but without "why" text.
    pass

# Build a SOC -> title mapping if present
soc_title_col = None
for candidate in ["soc_title", "Title", "onet_title"]:
    if candidate in df.columns:
        soc_title_col = candidate
        break

# If not present, use SOC code only
if soc_title_col is None:
    soc_titles = (
        df[["soc_code_final"]]
        .dropna()
        .drop_duplicates()
        .assign(soc_title=lambda x: x["soc_code_final"])
    )
else:
    soc_titles = (
        df[["soc_code_final", soc_title_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={soc_title_col: "soc_title"})
    )

soc_titles = soc_titles.rename(columns={"soc_code_final": "soc_code"}).sort_values("soc_code")

# Positions per user (for UI)
keep_cols = [
    "position_id", "user_id", "jobtitle_raw", "soc_code_final",
    "match_score_final", "match_score_tfidf_v2",
    "edu_match_score", "exp_match_score", "train_match_score",
    "match_category"
]
keep_cols = [c for c in keep_cols if c in df.columns]

pos_small = df[keep_cols].copy()
pos_small = pos_small.rename(columns={"soc_code_final": "soc_code"})

# Users list (for dropdown)
user_summary = (
    pos_small.groupby("user_id")
    .agg(
        n_positions=("position_id", "count"),
        avg_match=("match_score_final", "mean"),
        latest_title=("jobtitle_raw", lambda s: s.dropna().astype(str).iloc[-1] if len(s.dropna()) else "")
    )
    .reset_index()
    .sort_values(["avg_match", "n_positions"], ascending=[False, False])
)

# Validation stats (hardcode from your results or recompute here if you want)
# We'll store the headline metrics you already computed.
metrics = {
    "share_missing_soc_code_final": float(df["soc_code_final"].isna().mean()) if "soc_code_final" in df.columns else None,
    "final_score_mean": float(df["match_score_final"].mean()),
    "final_score_std": float(df["match_score_final"].std()),
    "improvement_auc": 0.705678329312622,  # <-- your result
    "improvement_decile_low": 0.83945,
    "improvement_decile_high": 0.082569
}

# Recommendations
# If you already exported recs_by_user.json elsewhere, skip this section.
# We'll do a simple, lightweight recommender using precomputed columns if present:
# - If user_doc and soc_doc exist, we compute char-gram TFIDF similarity and top-k SOCs per user.
recs_by_user: dict[str, list[dict]] = {}

if "user_doc" in df.columns and "soc_doc" in df.columns:
    from sklearn.feature_extraction.text import TfidfVectorizer

    users = (
        df.groupby("user_id")["user_doc"]
        .first()
        .fillna("")
        .reset_index()
    )
    users = users[users["user_doc"].str.len() > 0].reset_index(drop=True)

    jobs = (
        df.groupby("soc_code_final")["soc_doc"]
        .first()
        .fillna("")
        .reset_index()
        .rename(columns={"soc_code_final": "soc_code"})
    )
    jobs = jobs[jobs["soc_doc"].str.len() > 0].reset_index(drop=True)

    # Join titles onto jobs
    jobs = jobs.merge(soc_titles, on="soc_code", how="left")
    jobs["soc_title"] = jobs["soc_title"].fillna(jobs["soc_code"])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000)
    X_users = vec.fit_transform(users["user_doc"])
    X_jobs = vec.transform(jobs["soc_doc"])

    # Compute top-k efficiently in chunks
    TOPK = 10
    for i in range(X_users.shape[0]):
        # cosine similarity for L2-normalized tfidf = dot product
        scores = (X_jobs @ X_users[i].T).toarray().ravel()
        if len(scores) == 0:
            continue
        top_idx = np.argpartition(-scores, min(TOPK, len(scores))-1)[:TOPK]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        uid = str(users.loc[i, "user_id"])
        recs = []
        for j in top_idx:
            recs.append({
                "soc_code": str(jobs.loc[j, "soc_code"]),
                "soc_title": str(jobs.loc[j, "soc_title"]),
                "score": float(scores[j])
            })
        recs_by_user[uid] = recs

# ------------ write JSON artifacts ------------
def dump_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

dump_json(metrics, OUTDIR / "metrics.json")
dump_json(soc_titles.to_dict(orient="records"), OUTDIR / "soc_titles.json")
dump_json(user_summary.to_dict(orient="records"), OUTDIR / "users.json")

# positions grouped by user for easy UI lookup
pos_by_user = {}
for uid, g in pos_small.groupby("user_id"):
    pos_by_user[str(uid)] = g.sort_values("position_id").to_dict(orient="records")
dump_json(pos_by_user, OUTDIR / "positions_by_user.json")

dump_json(recs_by_user, OUTDIR / "recs_by_user.json")

print("Wrote demo artifacts to:", OUTDIR)
