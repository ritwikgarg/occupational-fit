from __future__ import annotations
from pathlib import Path
import json
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

MAX_TEXT_LENGTH = 10_000  # guard against oversized payloads
MAX_TOPK = 50


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "final_job_match_dataset.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"

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

# Load human-readable SOC titles from prebuilt JSON
soc_titles_path = DATA_DIR / "soc_titles.json"
if soc_titles_path.exists():
    _soc_titles = json.loads(soc_titles_path.read_text(encoding="utf-8"))
    _title_map = pd.DataFrame(_soc_titles).rename(columns={"soc_code": "soc_code", "soc_title": "soc_title"})
    jobs = jobs.merge(_title_map, on="soc_code", how="left")
    jobs["soc_title"] = jobs["soc_title"].fillna(jobs["soc_code"])
else:
    # Fallback: try columns in the dataset
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

@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def recommend():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip().lower()[:MAX_TEXT_LENGTH]
    topk = min(int(payload.get("topk", 10)), MAX_TOPK)
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


@app.route("/api/score", methods=["POST", "OPTIONS"])
def score():
    """Score arbitrary text against a specific SOC code."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip().lower()[:MAX_TEXT_LENGTH]
    soc_code = (payload.get("soc_code") or "").strip()

    if not text or not soc_code:
        return jsonify({"error": "Both 'text' and 'soc_code' are required"}), 400

    match = jobs[jobs["soc_code"] == soc_code]
    if match.empty:
        return jsonify({"error": f"SOC code '{soc_code}' not found"}), 404

    j = match.index[0]
    x = vec.transform([text])
    similarity = float((X_jobs[j] @ x.T).toarray().ravel()[0])

    return jsonify({
        "soc_code": soc_code,
        "soc_title": str(jobs.loc[j, "soc_title"]),
        "similarity": similarity,
    })


@app.route("/api/user/<user_id>", methods=["GET"])
def get_user(user_id):
    """Return user evidence and positions from prebuilt JSON artifacts."""
    # Load from demo data files
    evidence_path = DATA_DIR / "user_evidence.json"
    positions_path = DATA_DIR / "positions_by_user.json"

    result = {"user_id": user_id}

    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        result["evidence"] = evidence.get(str(user_id))
    else:
        result["evidence"] = None

    if positions_path.exists():
        positions = json.loads(positions_path.read_text(encoding="utf-8"))
        result["positions"] = positions.get(str(user_id))
    else:
        result["positions"] = None

    if result["evidence"] is None and result["positions"] is None:
        return jsonify({"error": f"User '{user_id}' not found"}), 404

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=8001, debug=True)
