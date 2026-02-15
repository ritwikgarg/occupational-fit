#!/usr/bin/env python3
"""Compute data-driven component weights from the existing dataset.

This is a standalone script that reads data/final_job_match_dataset.csv,
computes ANOVA eta-squared for each component, normalises to weights,
saves data/derived_weights.json, and re-scores all rows.

Run this INSTEAD of re-executing the full preprocessing notebook if the
dataset already exists and you only need to (re)derive the weights.

Usage:
    python compute_weights.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CSV  = DATA / "final_job_match_dataset.csv"
OUT  = DATA / "derived_weights.json"

COMPONENTS = [
    "match_score_tfidf_v2",
    "edu_match_score",
    "exp_match_score",
    "train_match_score",
]
LABELS = ["tfidf", "edu", "exp", "train"]


def main():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found. Run preprocessing.ipynb first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {CSV.name}")

    # Fill missing component scores conservatively
    for c in COMPONENTS[1:]:
        df[c] = df[c].fillna(0.5)

    # Keep rows with mapped SOC
    df_eta = df.dropna(subset=["soc_code_final"]).copy()
    print(f"Using {len(df_eta)} rows with valid SOC for eta-squared ({df_eta['soc_code_final'].nunique()} unique SOCs)")

    # ── Eta-squared ────────────────────────────────────────────
    eta_sq = {}
    for comp in COMPONENTS:
        grand_mean = df_eta[comp].mean()
        grp = df_eta.groupby("soc_code_final")[comp].agg(["mean", "count"])
        ss_between = ((grp["mean"] - grand_mean) ** 2 * grp["count"]).sum()
        ss_total = ((df_eta[comp] - grand_mean) ** 2).sum()
        eta_sq[comp] = ss_between / ss_total if ss_total > 0 else 0.0

    print("\n--- Eta-squared (between-SOC variance / total variance) ---")
    for comp, eta in eta_sq.items():
        bar = "█" * int(eta * 50)
        print(f"  {comp:25s}  η² = {eta:.4f}  {bar}")

    # ── Normalise to weights ───────────────────────────────────
    raw = np.array([eta_sq[c] for c in COMPONENTS])
    weights = raw / raw.sum()
    wdict = dict(zip(LABELS, [round(float(w), 4) for w in weights]))

    print("\n--- Derived weights ---")
    for label, w in wdict.items():
        print(f"  w_{label} = {w:.4f}")
    print(f"  Sum = {sum(wdict.values()):.4f}")

    # ── Re-score ───────────────────────────────────────────────
    w_tfidf, w_edu, w_exp, w_train = weights
    df["match_score_final"] = (
        w_tfidf * df["match_score_tfidf_v2"]
        + w_edu   * df["edu_match_score"]
        + w_exp   * df["exp_match_score"]
        + w_train * df["train_match_score"]
    )
    print(f"\nmatch_score_final stats:\n{df['match_score_final'].describe()}")

    # ── Save weights JSON ──────────────────────────────────────
    payload = {
        "method": "eta_squared_anova",
        "description": (
            "Weights derived from between-SOC variance ratio (eta-squared). "
            "Higher eta-squared means the component better discriminates "
            "between occupations."
        ),
        "weights": {
            "match_score_tfidf_v2": wdict["tfidf"],
            "edu_match_score": wdict["edu"],
            "exp_match_score": wdict["exp"],
            "train_match_score": wdict["train"],
        },
        "eta_squared": {c: round(float(eta_sq[c]), 6) for c in COMPONENTS},
        "n_soc_groups": int(df_eta["soc_code_final"].nunique()),
        "n_positions": len(df_eta),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved weights → {OUT}")

    # ── Re-export CSV ──────────────────────────────────────────
    df.to_csv(CSV, index=False)
    print(f"Updated scores → {CSV}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
