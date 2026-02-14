from __future__ import annotations
from pathlib import Path
import json
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "final_job_match_dataset.csv"

df = pd.read_csv(DATASET)
if "soc_doc" not in df.columns:
    raise RuntimeError("final_job_match_dataset.csv must include soc_doc for custom recommend API")

jobs = (
    df.groupby("soc_code_final")["soc_doc"]
    .first()
    .fillna("")
    .reset_index()
    .rename(columns={"soc_code_final": "soc_code"})
)
jobs = jobs[jobs["soc_doc"].str.len() > 0].reset_index(drop=True)

# Try to get titles if present
title_col = None
for cand in ["soc_title", "Title", "onet_title"]:
    if cand in df.columns:
        title_col = cand
        break
if title_col:
    titles = df[["soc_code_final", title_col]].dropna().drop_duplicates().rename(
        columns={"soc_code_final": "soc_code", title_col: "soc_title"}
    )
    jobs = jobs.merge(titles, on="soc_code", how="left")
jobs["soc_title"] = jobs.get("soc_title", jobs["soc_code"]).fillna(jobs["soc_code"])

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=200_000)
X_jobs = vec.fit_transform(jobs["soc_doc"])

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/recommend")
def recommend():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip().lower()
    topk = int(payload.get("topk", 10))
    if not text:
        return jsonify({"recs": []})

    x = vec.transform([text])
    scores = (X_jobs @ x.T).toarray().ravel()

    k = min(topk, len(scores))
    idx = np.argpartition(-scores, k-1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    recs = []
    for j in idx:
        recs.append({
            "soc_code": str(jobs.loc[j, "soc_code"]),
            "soc_title": str(jobs.loc[j, "soc_title"]),
            "score": float(scores[j])
        })
    return jsonify({"recs": recs})

if __name__ == "__main__":
    app.run(port=8001, debug=True)
