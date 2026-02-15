# Measuring Occupational Fit: Approach and Implementation

## The Problem

The goal here is to construct a measure of how well a worker fits a given job. We have O\*NET data describing what occupations require (tasks, education, experience, training) and sample data on individuals — their employment history, education, and self-reported skills. The question is: can we combine these to produce a meaningful "fit score"?

I'll walk through how I approached this step by step, what I tried, what worked, what didn't, and what I'd do differently with more time and data.

---

## Step 1: The First Challenge — Linking Jobs to Occupations

Before I could measure anything, I had to solve a practical problem: the individual data uses free-text job titles (`jobtitle_raw` — things like "Sr. Data Analyst" or "admin assistant"), but O\*NET organizes everything by standardized SOC codes (like `15-2051.00, Data Scientists`). I needed to map one to the other.

O\*NET helpfully provides two lookup files — **Alternate Titles** and **Sample of Reported Titles** — which list various names people use for each occupation. So I built a three-stage matching pipeline:

1. **Exact match** — normalize the job title (lowercase, strip punctuation and extra whitespace), then look it up directly in the combined alternate/reported titles. About 40% of positions matched this way.

2. **Fuzzy match** — for titles that didn't match exactly, I used RapidFuzz (a fast string-similarity library) to find the closest O\*NET title. I set a threshold of 90 (on a 0–100 scale) to avoid bad matches. For non-English titles, I raised this to 97 since character-level fuzzy matching across languages is unreliable.

3. **Fallback** — anything that still didn't match got assigned the closest available SOC code, but flagged with a low confidence score (0.3) so downstream steps know to be cautious.

I also compute a **mapping confidence** score (0–1) for each position. Exact matches get 1.0, fuzzy matches get a scaled score based on how similar the strings were, and fallbacks get 0.3. This lets anyone using the data see at a glance how trustworthy the SOC assignment is.

---

## Step 2: Building the Skill/Task Similarity Score

This was the most interesting part. The idea: if someone's profile text (their skills, job descriptions, education) talks about the same things as the O\*NET task descriptions for their occupation, they're probably a good fit.

For each SOC code, I concatenated all its O\*NET task descriptions (weighted by importance) into a single "task document." For each user, I built a "profile document" from their skills, education fields, and job title text.

Then I used **TF-IDF with cosine similarity** to compare them. Specifically, I used character n-grams (3 to 5 characters) instead of whole words. This was a deliberate choice — character n-grams handle compound words, abbreviations, and even some cross-language terms better than word-level tokenization. For example, "mgmt" and "management" share character n-grams even though they're different words.

The resulting similarity score (0–1) became the `match_score_tfidf_v2` component.

**An honest caveat**: TF-IDF captures surface-level lexical overlap, but it doesn't truly understand semantics. "Data analysis" and "statistical modeling" are conceptually very similar but don't share many characters. For a production system, I'd use **semantic embeddings** — something like a sentence transformer (e.g., all-MiniLM-L6) that maps text into a space where meaning matters, not just character overlap. TF-IDF was the right choice for prototyping because it's fast, interpretable, and doesn't require GPU or pre-trained model downloads, but it has a ceiling.

---

## Step 3: Education, Experience, and Training Gaps

The O\*NET data also tells us what education level, experience, and training each occupation typically requires. I mapped these to standardized scales:

- **Education**: I mapped degree strings from the individual data (things like "Bachelor of Science," "MBA," "High School Diploma") to O\*NET's Required Level scale (1–12, from "Less than High School" up to "Post-Doctoral Training"). This mapping uses regex patterns and covers 11 degree categories. Then the education gap is simply: user's level minus required level.

- **Experience**: I computed each user's cumulative prior work experience in years (from their employment history), converted that to O\*NET's experience-level categories, and compared against what the occupation requires.

- **Training**: Similar approach — comparing a proxy of the user's training level against the O\*NET requirement.

For each of these, I converted the gap into a 0–1 score using `clip(1 + gap/4, 0, 1)`. This means meeting or exceeding the requirement gives a score of 1.0, and being increasingly short of it pulls the score toward 0.

---

## Step 4: Combining Components — Why Not Just Pick Weights?

At first I used fixed weights for the composite score: 60% task similarity, 20% education, 15% experience, 5% training. These felt reasonable — skills should matter most — but they were arbitrary.

So I replaced them with **data-derived weights** using ANOVA eta-squared (η²). The idea is simple: for each component, I computed how much of its total variance is explained by which occupation (SOC code) a position belongs to. Components that vary a lot *between* occupations but not much *within* them are stronger signals of occupational fit, so they should get higher weight.

The math: for each component, η² = SS_between / SS_total, where SS_between measures across-SOC-group variance and SS_total is total variance. Then I normalized the four η² values to sum to 1.

The derived weights came out to:

| Component | η² | Weight |
|-----------|-----|--------|
| Task/skill similarity (TF-IDF) | 0.584 | 29.2% |
| Education match | 0.310 | 15.5% |
| Experience match | 0.338 | 16.9% |
| Training match | 0.772 | 38.5% |

Interestingly, training had the highest discriminative power across occupations — probably because training requirements vary a lot between occupation types (a surgeon vs. a truck driver). Task similarity was second. This is a more principled weighting than just hardcoding numbers.

The final composite score is:

$$\text{match\_score\_final} = w_{\text{tfidf}} \times \text{tfidf} + w_{\text{edu}} \times \text{edu} + w_{\text{exp}} \times \text{exp} + w_{\text{train}} \times \text{train}$$

where the weights $w$ are the η²-derived values above.

---

## Step 5: Validating the Score

A score is meaningless if it doesn't actually predict anything. I ran several validation checks:

1. **Formula verification** — recompute the score from components and confirm it matches, to rule out pipeline bugs.

2. **Sanity checks** — exact SOC matches should score higher than fuzzy matches, which should score higher than fallbacks. They do.

3. **Shuffle test** — randomly shuffle the SOC assignments and recompute. If the score structure survives shuffling, it's not capturing real signal. It doesn't survive — shuffled scores are significantly worse.

4. **Predictive validation** — I trained a Random Forest classifier (5-fold stratified cross-validation) to predict two things:
   - Whether someone *switched occupations* at their next job (yes = moved to a different SOC code)
   - Whether their match score *improved* at the next job

   The **improvement AUC was 0.71**, meaning the score has real predictive power: people with lower fit scores tend to move toward better-fitting roles in their next job. That's a meaningful signal for a first-pass model.

5. **Feature importance** — both Gini importance and permutation importance confirm that the task similarity and training components are the strongest predictors, consistent with the η² weighting.

---

## Step 6: A Recommender

As a natural extension, I built a simple occupation recommender. Given a user's profile text, it finds the most similar SOC codes using the same TF-IDF setup. It filters out occupations the user has already held (so recommendations are actionable), and for each recommendation it can show which text features drove the match — making it somewhat interpretable.

---

## Step 7: The Demo UI

I wanted all of this to be inspectable and tangible, not just numbers in a notebook. So I built a small interactive dashboard.

The UI is a single HTML page that talks to a lightweight Flask API. Here's what it shows:

- **User picker** — select any user from the dataset
- **Position table** — see all positions that user has held, with their SOC code, fit score, and a label (well-matched, underqualified, overqualified, or mismatch). Click any position to inspect it.
- **Overall fit score** — a large score display with a color-coded progress bar (green = good fit, red = poor fit)
- **Component breakdown** — mini-bars showing the task similarity, education, experience, and training scores individually, with the data-derived weights displayed as percentages
- **User evidence panel** — shows *what data went into the score*: the user's skills/profile text, their education level, and the occupation's requirements from O\*NET. This is important for transparency — you can see exactly why someone scored the way they did.
- **"Why this score" explanation** — a plain-English summary of what's driving the score up or down
- **Mapping confidence badge** — color-coded indicator of how reliably the job title was mapped to a SOC code
- **Recommendations table** — suggested alternative occupations based on profile similarity, with human-readable titles
- **Custom API query** — paste any text and score it against any SOC code, or get top-K recommendations. This lets you test the system with arbitrary input without needing to be in the dataset.

The Flask API exposes four endpoints: health check, recommend (score text against all SOCs), score (score text against a specific SOC), and user lookup (retrieve a user's evidence and positions).

---

## What I'd Do With More Data and Time

A few things I'd prioritize if this were going to production:

1. **Semantic embeddings instead of TF-IDF** — a sentence transformer would capture meaning, not just surface overlap. "Financial analyst" and "investment researcher" would match better.

2. **Better training/experience scoring** — right now these are coarse gap calculations. With more data, I'd train a model that learns the nonlinear relationship between experience and fit.

3. **Temporal dynamics** — the current score is a snapshot. With enough data, you could model how fit changes over time as someone gains experience.

4. **True exit labels** — right now `job_switch` is a proxy (1 for every non-last position). With real voluntary-exit data, the predictive model would be much stronger.

5. **Non-English support** — character n-gram TF-IDF partially handles this, but a multilingual embedding model would be far better.

---

## How to Run

The code is on GitHub. To reproduce:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the preprocessing pipeline
#    Open notebooks/preprocessing.ipynb and run all cells.
#    This generates data/final_job_match_dataset.csv and data/derived_weights.json.

# 3. (Optional) Run validation and predictive model notebooks
#    notebooks/validation.ipynb — sanity checks
#    notebooks/predictive_model.ipynb — AUC and feature importance

# 4. Build demo artifacts
python demo/build_demo_artifacts.py

# 5. Start the API server and open the UI
python demo/app.py
# Then open demo/index.html in a browser
```

There's also a standalone `compute_weights.py` script that re-derives the component weights from the existing dataset without re-running the full notebook — useful for quick iteration.

---

## Summary

The approach is straightforward: map job titles to standardized occupations, compare worker profiles against occupation requirements across four dimensions (task similarity, education, experience, training), weight the components using their actual discriminative power from the data, and combine into a single 0–1 fit score. The score validly predicts career movement (AUC = 0.71), and the interactive demo makes it inspectable for any user in the dataset. The main limitation is TF-IDF's semantic ceiling — swapping in embeddings would be the single biggest improvement.
