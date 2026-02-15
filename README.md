# Occupational Fit

**Construct a measure of the quality of matches between jobs and workers.**

This project builds a composite occupational-fit score by combining:
- **Task/skill similarity** (TF-IDF character n-gram cosine similarity between user profile text and O\*NET task descriptions)
- **Education gap** (user education level vs. O\*NET required education level)
- **Experience gap** (user career experience vs. O\*NET required experience)
- **Training match** (user training proxy vs. O\*NET required training)

The final score is:

$$\text{match\_score\_final} = w_{\text{tfidf}} \times \text{tfidf} + w_{\text{edu}} \times \text{edu} + w_{\text{exp}} \times \text{exp} + w_{\text{train}} \times \text{train}$$

Weights are **derived from data** using ANOVA eta-squared (η²): for each component, we compute the ratio of between-SOC variance to total variance, then normalize so all weights sum to 1. This ensures components that better discriminate between occupations receive higher weight. The derived weights are saved to `data/derived_weights.json`.

---

## Project Structure

```
├── data/
│   ├── final_job_match_dataset.csv     # Main output: scored positions
│   ├── artifacts/                      # Intermediate pipeline outputs
│   └── sample_data_extracted/          # O*NET + individual source data
├── notebooks/
│   ├── preprocessing.ipynb             # Core pipeline (Steps 1–12)
│   ├── validation.ipynb                # Sanity checks & calibration
│   ├── predictive_model.ipynb          # External validation (AUC, feature importance)
│   └── recommender.ipynb              # SOC recommendation engine
├── demo/
│   ├── build_demo_artifacts.py         # Generate JSON files for the UI
│   ├── app.py                          # Flask API (recommend, score, user lookup)
│   └── index.html                      # Interactive demo dashboard
├── requirements.txt
└── README.md
```

## Data Sources

| Source | Contents |
|--------|----------|
| `individual_position.csv` | User employment history (title, dates, seniority) |
| `individual_user_education.csv` | User education records (degree, field, school) |
| `individual_user_skill.csv` | User skill lists |
| O\*NET Task Ratings | Task descriptions and importance ratings per SOC |
| O\*NET Education/Training/Experience | Required levels per SOC |
| O\*NET Alternate Titles | Canonical + alternate job titles per SOC |

## Pipeline Overview

### 1. Preprocessing (`preprocessing.ipynb`)
1. **Extract** sample data from ZIP
2. **Normalize** job titles (lowercase, strip punctuation/whitespace)
3. **Map to SOC** via exact match → fuzzy match (RapidFuzz, threshold 90) → fallback
4. **Build task documents** by concatenating O\*NET task descriptions weighted by importance
5. **Compute TF-IDF similarity** between user profile and SOC task documents
6. **Score education gap** against O\*NET required education level
7. **Score experience/training gaps** similarly
8. **Derive component weights** via ANOVA eta-squared (between-SOC variance ratio)
9. **Composite score** using derived weights
9. **Categorize** matches (well-matched, underqualified, overqualified, mismatch)
10. **Compute mapping confidence** (0–1) based on match type and quality
11. **Export** `final_job_match_dataset.csv` (51 columns)

### 2. Validation (`validation.ipynb`)
- Formula verification (recompute and compare)
- Match-type ordering (exact > fuzzy > fallback)
- Shuffle test (verify structure isn't random)
- Score distribution / calibration analysis
- Component correlation analysis

### 3. Predictive Model (`predictive_model.ipynb`)
- **Cross-validated Random Forest** predicting job switches (5-fold stratified)
- **Feature importance** (Gini + permutation-based)
- **Occupation switch AUC**: does low match predict changing SOC?
- **Match improvement AUC**: does low match predict higher score at next job?
- **Career trajectory**: match score trends across positions

### 4. Recommender (`recommender.ipynb`)
- TF-IDF cosine similarity (user profile → SOC task documents)
- Filters out user's current/past SOC codes
- Explains recommendations via top overlapping TF-IDF features
- Returns alternate job titles per recommended SOC

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the preprocessing pipeline
# Open notebooks/preprocessing.ipynb and run all cells

# Generate demo data
python demo/build_demo_artifacts.py

# Start the API server
python demo/app.py
# Then open demo/index.html in a browser
```

### Requirements
- Python 3.10+
- pandas, numpy, scikit-learn, rapidfuzz, openpyxl, flask

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/recommend` | Recommend SOCs for arbitrary profile text |
| `POST` | `/api/score` | Score text against a specific SOC code |
| `GET` | `/api/user/<id>` | Get user evidence and positions |

### Example: Score a profile against a SOC

```bash
curl -X POST http://127.0.0.1:8001/api/score \
  -H "Content-Type: application/json" \
  -d '{"text": "data analysis python sql machine learning", "soc_code": "15-2051.00"}'
```

## Key Design Decisions

1. **Character n-grams (3,5)** instead of word-level TF-IDF — handles compound words, abbreviations, and cross-language terms better
2. **Data-derived weights (ANOVA η²)** — component weights are determined by each feature's discriminative power across SOC codes, replacing earlier arbitrary fixed weights
3. **Mapping confidence** — tracks how reliably a job title was mapped to a SOC code (exact=1.0, fuzzy=scaled, fallback=0.3)
4. **Past-SOC filtering** in recommendations — ensures suggested occupations are actionable (not already held)

## Known Limitations

- `job_switch` target is a proxy (1 for all non-last positions) — not a true voluntary-exit indicator
- Education level mapping covers common degrees but may miss non-standard credentials
- Training score relies on coarse O\*NET categories
- SOC mapping quality depends on job title normalization; non-English titles receive a confidence penalty
- Sample size limits statistical power for career-trajectory analyses

## Output Dataset Schema

The main output `final_job_match_dataset.csv` contains 51 columns including:

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `position_id` | Position identifier |
| `jobtitle_raw` / `jobtitle_norm` | Original and normalized job title |
| `soc_code_final` | Mapped O\*NET SOC code |
| `match_type` | How the SOC was matched (exact/fuzzy/fallback) |
| `match_score_final` | Composite fit score (0–1) |
| `match_score_tfidf_v2` | Task/skill similarity component |
| `edu_match_score` | Education match component |
| `exp_match_score` | Experience match component |
| `train_match_score` | Training match component |
| `match_category` | Label (well-matched/underqualified/overqualified/mismatch) |
| `mapping_confidence` | Confidence in SOC code mapping (0–1) |
| `user_doc` | User profile text used for TF-IDF |
| `soc_doc` | SOC task description text |