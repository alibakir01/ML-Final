# Forecasting Quarterly Short-Term-Rental Demand — Journal Submission

This repository holds the code and manuscript for a journal submission on
NYC short-term-rental demand forecasting, built on **public, open Inside
Airbnb data (2025–2026)**.

The project originated as a COMP 468 (Introduction to Machine Learning, AGU)
course project on a closed Kaggle dataset — that coursework is complete and
graded. The supervisor then offered the opportunity to extend it into a
publishable study as part of an **AGU internship program**; the pipeline was
re-built end-to-end on open data so the results and code can be shared with
journal editors/reviewers without a proprietary dataset. A teammate
voluntarily contributed to this extension outside of the internship itself.

**`inside_airbnb/` is the code path for this submission — start there.**

---

## Manuscript

*"Forecasting Quarterly Short-Term-Rental Demand from Public Availability
Snapshots: A Leakage-Safe Convex Ensemble on 2025–2026 New York City Data"*

- Source: [`inside_airbnb/paper2/main_p2.tex`](inside_airbnb/paper2/main_p2.tex)
- Built manuscript: [`inside_airbnb/paper2/Paper2_Airbnb_2025_2026.pdf`](inside_airbnb/paper2/Paper2_Airbnb_2025_2026.pdf) / `.docx`
- Draft results narrative: [`inside_airbnb/RESULTS_SECTION.md`](inside_airbnb/RESULTS_SECTION.md)
- Plain-language data guide (what every file/column is): [`inside_airbnb/readable/00_OVERVIEW.md`](inside_airbnb/readable/00_OVERVIEW.md)

**Data source:** publicly available [Inside Airbnb](http://insideairbnb.com)
NYC snapshots (`listings.csv`, `calendar.csv`, `reviews.csv`), monthly scrapes
from July 2025 through April 2026. No proprietary or competition data is used.

**Problem:** predict the number of nights a listing will be booked in a
future quarter, from its preceding calendar/listing history. The booking
label is recovered by differencing consecutive `available` snapshots
(a censored, zero-inflated count in `[0, 92]`), since Inside Airbnb does not
expose bookings directly.

**Headline result (Setup C — 7-month history → Feb–Mar–Apr 2026):**

| Model | Dev R² (5-fold CV) | Held-out Test R² | Test MSE |
|---|---|---|---|
| Linear Regression | 0.581 | 0.568 | 326.0 |
| Random Forest | 0.652 | 0.644 | 268.7 |
| XGBoost | 0.659 | 0.647 | 266.2 |
| LightGBM | 0.659 | 0.649 | 264.6 |
| CatBoost | 0.654 | 0.641 | 270.8 |
| **SLSQP convex blend (proposed)** | **0.663** | **0.653** | **262.0** |

Held-out test R² (0.653) tracks the CV estimate (0.663) within 0.01, and the
result is stable across three independent target quarters (see Table 2 in
`RESULTS_SECTION.md`). Full details: significance testing, ablation study, and
discussion of the data-driven error ceiling are in `RESULTS_SECTION.md` and
Section V of the manuscript.

## `inside_airbnb/` code map

| Stage | Scripts |
|---|---|
| Target construction | `target_q4_clean.py`, `target_q1_clean.py`, `target_setupC.py`, `target_occupancy_setupC.py`, `target_q4_booking_diff.py`, `fused_target.py` |
| Feature building | `build_features_setupA.py`, `build_setupB2Q.py`, `prep_setupB.py`, `prep_setupC_features.py`, `add_price_features.py` / `_v2.py`, `reviews_features.py`, `enhance_setupC.py`, `enhance_v3.py` |
| Model training / CV | `model_setupA.py`, `model_run.py`, `model_run_timed.py`, `tune.py`, `_mlp_cv_worker.py`, `tune_mlp_setupC.py`, `experiment_mlp_setupC.py` |
| Held-out evaluation | `heldout_setupC.py`, `heldout_setupC_with_mlp.py`, `heldout_hurdle_setupC.py`, `hurdle.py`, `_heldout_mlp_worker.py` |
| EDA / diagnostics | `eda_tour.py`, `eda_target_q4_2025.py`, `analyze_setup.py` |
| Human-readable docs | `make_readable.py`, `readable_2026.py` → populate `readable/` |
| Paper build | `paper2_figures.py` (regenerates all figures), `paper2/build_paper2.py` (LaTeX → IEEE-formatted Word/PDF) |

All intermediate artifacts (feature parquets, OOF predictions, result `.txt`
summaries) are written to `inside_airbnb/outputs/`.

**Dependencies:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`,
`lightgbm`, `catboost`, `torch`, `matplotlib`.

---

## Repository history / other branches

- **`notebooks/` + `paper/`** (this branch): the original graded COMP 468
  course project — predicting booked nights on the closed Kaggle *K353*
  competition (2016 NYC data). Kept for provenance; not part of the journal
  submission. Competition: <https://www.kaggle.com/competitions/a-cloned-airbnb-booking-prediction-competition-k-353>
- **`Internship/`** (on the `Migrate_2025/26_bashkal` branch, not on
  `main`/`Ali`): a first, independent notebook-by-notebook copy of the same
  migration to open data, with its own mirrored `paper/`. Superseded by
  `inside_airbnb/`, which produced the stronger pipeline and results used
  here.

## Acknowledgment
Originated as coursework in COMP 468 at **Abdullah Gül University (AGU)**;
extended for journal submission as part of an AGU internship program, with
voluntary help from a teammate.
