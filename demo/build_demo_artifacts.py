from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ------------ paths ------------
ROOT = Path(__file__).resolve().parents[1]  # project root if demo/ is under it
DATASET = ROOT / "data" / "final_job_match_dataset.csv"

# Extracted sample data (adjust if your folder differs)
EXTRACTED = ROOT / "data" / "sample_data_extracted" / "sample_data"
INDIV_DIR = EXTRACTED / "individual_data"
OCC_DIR = EXTRACTED / "occupation_requirements"

EDU_CSV = INDIV_DIR / "individual_user_education.csv"
SKILL_CSV = INDIV_DIR / "individual_user_skill.csv"

ALT_TITLES_XLSX = OCC_DIR / "Alternate Titles.xlsx"
REPORTED_TITLES_XLSX = OCC_DIR / "Sample of Reported Titles.xlsx"

OUTDIR = Path(__file__).resolve().parent / "data"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------ helpers ------------
def f(x):
    x = float(x)
    return None if (np.isnan(x) or np.isinf(x)) else x


def _json_sanitize(x):
    """Recursively convert NaN/Inf to None so json is valid."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (np.floating,)):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_sanitize(v) for v in x]
    if x is np.nan:
        return None
    return x


def dump_json(obj, path: Path):
    obj = _json_sanitize(obj)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def compute_tenure_months(df: pd.DataFrame) -> pd.Series:
    if "startdate" not in df.columns or "enddate" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    start = safe_dt(df["startdate"])
    end = safe_dt(df["enddate"])
    return (end - start).dt.days / 30.4


def years_between(a: pd.Series, b: pd.Series) -> pd.Series:
    # b - a in years
    return (b - a).dt.days / 365.25


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def req_years_from_score(score: float) -> float | None:
    """
    Turn exp_match_score (0-1) into an interpretable "required years" proxy.

    We don't know the exact O*NET RW category used, so we expose a calibrated proxy:
      - exp score near 1 => user meets/exceeds requirement => requirement <= user exp
      - exp score near 0 => far below => requirement >> user exp

    We'll map score -> required_years_multiplier:
      score 1.0 -> mult 0.8
      score 0.7 -> mult 1.0
      score 0.4 -> mult 1.6
      score 0.0 -> mult 2.5
    Then required_years = user_exp_years * mult, clipped to [0, 15].
    """
    if score is None or (isinstance(score, float) and (math.isnan(score) or math.isinf(score))):
        return None
    s = clamp01(score)
    # piecewise-ish smooth curve
    mult = 0.8 + (1.7 * (1.0 - s) ** 1.3)  # 0.8..2.5-ish
    return mult


def req_train_months_from_score(score: float) -> float | None:
    """
    Training score proxy -> "required months" proxy.

    Map (train_match_score) into approximate required training months:
      score 1.0 => low formal training requirement (<= 1 month)
      score 0.5 => medium (3-6 months)
      score 0.0 => high (12+ months)
    """
    if score is None or (isinstance(score, float) and (math.isnan(score) or math.isinf(score))):
        return None
    s = clamp01(score)
    # invert: lower score -> more training required
    months = 1.0 + (12.0 * (1.0 - s) ** 1.2)  # 1..13
    return months


def human_train_bucket(months: float | None) -> str | None:
    if months is None:
        return None
    if months <= 1.5:
        return "Up to 1 month"
    if months <= 3.5:
        return "1–3 months"
    if months <= 6.5:
        return "3–6 months"
    if months <= 12.5:
        return "6–12 months"
    return "12+ months"


# ------------ load dataset ------------
df = pd.read_csv(DATASET)

# Basic cleanup
for c in ["user_doc", "soc_doc"]:
    if c in df.columns:
        df[c] = df[c].fillna("")

# Standardize SOC key
if "soc_code_final" not in df.columns and "soc_code" in df.columns:
    df["soc_code_final"] = df["soc_code"]

# Ensure tenure_months exists
if "tenure_months" not in df.columns:
    df["tenure_months"] = compute_tenure_months(df)

# Parse startdate/enddate for experience features
if "startdate" in df.columns:
    df["startdate_dt"] = safe_dt(df["startdate"])
else:
    df["startdate_dt"] = pd.NaT

if "enddate" in df.columns:
    df["enddate_dt"] = safe_dt(df["enddate"])
else:
    df["enddate_dt"] = pd.NaT

# ------------ build SOC TITLES (human-readable) ------------
soc_titles_df = None

def build_soc_titles_from_onet() -> pd.DataFrame | None:
    # Prefer Alternate Titles.xlsx because it has canonical Title + Alternate Title
    if ALT_TITLES_XLSX.exists():
        alt = pd.read_excel(ALT_TITLES_XLSX)
        # expected cols: 'O*NET-SOC Code', 'Title', 'Alternate Title', ...
        cols = alt.columns.astype(str).tolist()
        code_col = "O*NET-SOC Code" if "O*NET-SOC Code" in cols else None
        title_col = "Title" if "Title" in cols else None
        if code_col and title_col:
            soc = (
                alt[[code_col, title_col]]
                .dropna()
                .drop_duplicates()
                .rename(columns={code_col: "soc_code", title_col: "soc_title"})
            )
            # In case multiple titles exist, keep the first per SOC
            soc = soc.groupby("soc_code", as_index=False)["soc_title"].first()
            return soc

    # Fallback: Sample of Reported Titles.xlsx
    if REPORTED_TITLES_XLSX.exists():
        rep = pd.read_excel(REPORTED_TITLES_XLSX)
        cols = rep.columns.astype(str).tolist()
        code_col = "O*NET-SOC Code" if "O*NET-SOC Code" in cols else None
        title_col = "Title" if "Title" in cols else None
        if code_col and title_col:
            soc = (
                rep[[code_col, title_col]]
                .dropna()
                .drop_duplicates()
                .rename(columns={code_col: "soc_code", title_col: "soc_title"})
            )
            soc = soc.groupby("soc_code", as_index=False)["soc_title"].first()
            return soc

    return None


soc_titles_df = build_soc_titles_from_onet()

# If we couldn't load O*NET titles, fall back to SOC codes only
if soc_titles_df is None:
    soc_titles_df = (
        df[["soc_code_final"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"soc_code_final": "soc_code"})
        .assign(soc_title=lambda x: x["soc_code"])
    )

soc_titles_df = soc_titles_df.dropna().drop_duplicates().sort_values("soc_code")

# ------------ user-level experience years ------------
# Define "user experience years" as: years between earliest startdate and latest enddate (or today-ish max date)
# using only rows with valid dates.
user_exp_years = (
    df.dropna(subset=["user_id"])
    .assign(_start=df["startdate_dt"], _end=df["enddate_dt"])
    .groupby("user_id")
    .agg(
        first_start=("_start", "min"),
        last_end=("_end", "max")
    )
    .reset_index()
)

# If last_end missing, approximate with max startdate (not great but better than empty)
user_exp_years["last_end"] = user_exp_years["last_end"].fillna(user_exp_years["first_start"])
user_exp_years["user_exp_years"] = years_between(user_exp_years["first_start"], user_exp_years["last_end"]).clip(lower=0)
user_exp_years = user_exp_years[["user_id", "user_exp_years"]]

df = df.merge(user_exp_years, on="user_id", how="left")

# ------------ derive interpretable requirement proxies ------------
# required experience years (proxy) derived from exp_match_score and user_exp_years
if "exp_match_score" in df.columns:
    mult = df["exp_match_score"].apply(req_years_from_score)
    df["req_exp_years"] = (df["user_exp_years"] * mult).clip(lower=0, upper=15)
    df["exp_gap"] = (df["user_exp_years"] - df["req_exp_years"])
else:
    df["req_exp_years"] = np.nan
    df["exp_gap"] = np.nan

# training requirement proxy derived from train_match_score
if "train_match_score" in df.columns:
    df["req_training_months"] = df["train_match_score"].apply(req_train_months_from_score)
    df["req_training_level"] = df["req_training_months"].apply(human_train_bucket)
else:
    df["req_training_months"] = np.nan
    df["req_training_level"] = None

# user training proxy (very rough): if train score high assume meets requirement; if low assume not.
# We show it as a bucket purely for interpretability.
df["user_training_level"] = np.where(
    df["train_match_score"].fillna(0) >= 0.7, "Meets typical training requirement",
    np.where(df["train_match_score"].fillna(0) >= 0.4, "Partially meets training requirement", "Likely below training requirement")
)

# train gap proxy: positive means "user likely meets/exceeds"
df["train_gap"] = df["train_match_score"].fillna(np.nan) - 0.7

# ------------ build USER EVIDENCE (raw + normalized) ------------
user_evidence: dict[str, dict] = {}

# user_doc
if "user_doc" in df.columns:
    tmp = df.groupby("user_id")["user_doc"].first()
    for uid, doc in tmp.items():
        user_evidence.setdefault(str(uid), {})["user_doc"] = "" if pd.isna(doc) else str(doc)

# user_edu_level if present
for cand in ["user_edu_level", "edu_level_rl", "user_education_level"]:
    if cand in df.columns:
        tmp = df.groupby("user_id")[cand].max()
        for uid, v in tmp.items():
            user_evidence.setdefault(str(uid), {})["user_edu_level"] = None if pd.isna(v) else float(v)
        break

# user_exp_years
tmp = df.groupby("user_id")["user_exp_years"].max()
for uid, v in tmp.items():
    user_evidence.setdefault(str(uid), {})["user_exp_years"] = None if pd.isna(v) else float(v)

# raw education examples (if available)
if EDU_CSV.exists():
    edu = pd.read_csv(EDU_CSV)
    cols = [c for c in ["user_id", "school", "degree_raw", "field_raw", "startdate", "enddate"] if c in edu.columns]
    edu = edu[cols].copy()

    def fmt_edu_row(r) -> str:
        school = str(r.get("school", "") or "").strip()
        degree = str(r.get("degree_raw", "") or "").strip()
        field = str(r.get("field_raw", "") or "").strip()
        bits = [b for b in [degree, field] if b]
        left = " — ".join(bits) if bits else "(degree not listed)"
        return f"{left} @ {school}" if school else left

    edu["edu_line"] = edu.apply(fmt_edu_row, axis=1)
    edu_lines = (
        edu.groupby("user_id")["edu_line"]
        .apply(lambda s: "; ".join(list(dict.fromkeys([x for x in s if str(x).strip()]))[:8]))
    )
    for uid, line in edu_lines.items():
        user_evidence.setdefault(str(uid), {})["education_examples"] = "" if pd.isna(line) else str(line)

# skills (if available)
if SKILL_CSV.exists():
    skills = pd.read_csv(SKILL_CSV)
    skill_col = "skill_mapped" if "skill_mapped" in skills.columns else ("skill_raw" if "skill_raw" in skills.columns else None)
    if skill_col:
        top_skills = (
            skills.dropna(subset=["user_id"])
            .assign(_skill=lambda x: x[skill_col].astype(str).str.strip())
            .query("_skill != ''")
            .groupby("user_id")["_skill"]
            .apply(lambda s: list(dict.fromkeys(s.tolist()))[:30])
        )
        for uid, lst in top_skills.items():
            user_evidence.setdefault(str(uid), {})["skills"] = lst

# ------------ positions per user (for UI) ------------
BASE_COLS = [
    "position_id", "user_id",
    "jobtitle_raw", "mapped_role", "job_category", "seniority",
    "startdate", "enddate", "tenure_months",
    "soc_code_final", "match_type",
    "match_score_final", "match_score_tfidf_v2",
    "edu_match_score", "exp_match_score", "train_match_score",
    "match_category",

    # NEW: interpretable proxies
    "user_exp_years", "req_exp_years", "exp_gap",
    "req_training_level", "user_training_level", "train_gap",

    # include if you already have these in df
    "user_edu_level", "req_edu_level", "edu_gap",

    # Mapping confidence from preprocessing
    "mapping_confidence",
]

keep_cols = [c for c in BASE_COLS if c in df.columns]

pos_small = df[keep_cols].copy()
pos_small = pos_small.rename(columns={"soc_code_final": "soc_code"})

# ------------ Users list (dropdown) ------------
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

# ------------ metrics (computed dynamically) ------------

# Compute improvement AUC dynamically instead of hard-coding
improvement_auc = None
improvement_decile_low = None
improvement_decile_high = None

try:
    from sklearn.metrics import roc_auc_score

    _df_m = df.copy()
    _df_m["startdate_dt"] = safe_dt(_df_m["startdate"])
    _df_m = _df_m.sort_values(["user_id", "startdate_dt", "position_id"])
    _df_m["next_match"] = _df_m.groupby("user_id")["match_score_final"].shift(-1)
    _df_m["improved"] = (_df_m["next_match"] > _df_m["match_score_final"]).astype(int)
    _valid = _df_m.dropna(subset=["next_match", "match_score_final"])

    if len(_valid) > 10 and _valid["improved"].nunique() == 2:
        improvement_auc = float(roc_auc_score(_valid["improved"], -_valid["match_score_final"]))
        # Decile extremes
        _valid["_bin"] = pd.qcut(_valid["match_score_final"], 10, labels=False, duplicates="drop")
        _dec = _valid.groupby("_bin")["improved"].mean()
        improvement_decile_low = float(_dec.iloc[0]) if len(_dec) > 0 else None
        improvement_decile_high = float(_dec.iloc[-1]) if len(_dec) > 0 else None
        print(f"Computed improvement AUC: {improvement_auc:.4f}")
    else:
        print("Not enough data to compute improvement AUC")
except Exception as e:
    print(f"Could not compute improvement AUC: {e}")

metrics = {
    "share_missing_soc_code_final": f(df["soc_code_final"].isna().mean()) if "soc_code_final" in df.columns else None,
    "final_score_mean": f(df["match_score_final"].mean()) if "match_score_final" in df.columns else None,
    "final_score_std": f(df["match_score_final"].std()) if "match_score_final" in df.columns else None,
    "improvement_auc": improvement_auc,
    "improvement_decile_low": improvement_decile_low,
    "improvement_decile_high": improvement_decile_high,
}

# Load derived weights and include in metrics
WEIGHTS_PATH = ROOT / "data" / "derived_weights.json"
if WEIGHTS_PATH.exists():
    _w = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    metrics["weights"] = _w["weights"]
    metrics["weights_method"] = _w["method"]
    metrics["weights_eta_squared"] = _w.get("eta_squared")
    print(f"Loaded derived weights: {_w['weights']}")
else:
    # Fallback to hardcoded if file not found
    metrics["weights"] = {
        "match_score_tfidf_v2": 0.60,
        "edu_match_score": 0.20,
        "exp_match_score": 0.15,
        "train_match_score": 0.05,
    }
    metrics["weights_method"] = "hardcoded_fallback"
    print("WARNING: derived_weights.json not found, using fallback weights")

# ------------ Recommendations ------------
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

    # Join human-readable titles from O*NET
    jobs = jobs.merge(soc_titles_df, on="soc_code", how="left")
    jobs["soc_title"] = jobs["soc_title"].fillna(jobs["soc_code"])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000)
    X_users = vec.fit_transform(users["user_doc"])
    X_jobs = vec.transform(jobs["soc_doc"])

    TOPK = 10
    TOPK_CANDIDATES = TOPK + 20  # generate extra candidates to account for past-SOC filtering
    for i in range(X_users.shape[0]):
        scores = (X_jobs @ X_users[i].T).toarray().ravel()
        if len(scores) == 0:
            continue
        n_cand = min(TOPK_CANDIDATES, len(scores))
        top_idx = np.argpartition(-scores, n_cand - 1)[:n_cand]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        uid = str(users.loc[i, "user_id"])

        # Collect user's past SOC codes to filter them out
        user_past = set(
            df.loc[df["user_id"] == users.loc[i, "user_id"], "soc_code_final"]
            .dropna().astype(str).tolist()
        )

        recs = []
        for j in top_idx:
            soc = str(jobs.loc[j, "soc_code"])
            if soc in user_past:
                continue  # skip SOCs the user already held
            recs.append({
                "soc_code": soc,
                "soc_title": str(jobs.loc[j, "soc_title"]),
                "score": float(scores[j])
            })
        recs_by_user[uid] = recs[:TOPK]  # ensure we don't exceed TOPK after filtering

# ------------ write JSON artifacts ------------
dump_json(metrics, OUTDIR / "metrics.json")
dump_json(soc_titles_df.to_dict(orient="records"), OUTDIR / "soc_titles.json")
dump_json(user_summary.to_dict(orient="records"), OUTDIR / "users.json")

# positions grouped by user
pos_by_user: dict[str, list[dict]] = {}
for uid, g in pos_small.groupby("user_id"):
    pos_by_user[str(uid)] = g.sort_values("position_id").to_dict(orient="records")
dump_json(pos_by_user, OUTDIR / "positions_by_user.json")

dump_json(recs_by_user, OUTDIR / "recs_by_user.json")
dump_json(user_evidence, OUTDIR / "user_evidence.json")

print("Wrote demo artifacts to:", OUTDIR)
print("Artifacts:", [p.name for p in sorted(OUTDIR.glob("*.json"))])
print("SOC titles source:",
      "Alternate Titles.xlsx" if ALT_TITLES_XLSX.exists() else
      ("Sample of Reported Titles.xlsx" if REPORTED_TITLES_XLSX.exists() else "FALLBACK (SOC codes)"))
