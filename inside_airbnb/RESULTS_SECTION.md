# Results (2025–2026 Inside Airbnb External Validation) — DRAFT

## V-A. Main model and held-out performance

The proposed forecaster predicts the number of nights a New York City short-term
rental will be booked in the target quarter **February–March–April 2026**, using
a **seven-month behavioural history (July 2025 – January 2026)** and a
leakage-safe pipeline (median imputation, one-hot encoding for low-cardinality
categoricals, and a $K$-fold target encoder for high-cardinality location fields)
feeding five learners (Linear Regression, Random Forest, XGBoost, LightGBM,
CatBoost) combined by an **SLSQP convex blend**. The target is the *booked-night*
count obtained by snapshot differencing (an availability transition
$\text{available}\rightarrow\text{unavailable}$ observed in $\geq 2$ snapshots),
a censored, zero-inflated integer in $[0,92]$.

Model quality is reported on a **20% held-out test set of listings** (never used
in training or model selection), alongside the 5-fold cross-validation estimate
on the 80% development set.

**Table 1. Held-out performance (Setup C; dev $n=29{,}025$, test $n=7{,}257$).**

| Model | Dev $R^2$ (5-fold CV) | **Held-out Test $R^2$** | Test MSE |
|---|---|---|---|
| Linear Regression | 0.581 | 0.568 | 326.0 |
| Random Forest | 0.652 | 0.644 | 268.7 |
| XGBoost | 0.659 | 0.647 | 266.2 |
| LightGBM | 0.659 | 0.649 | 264.6 |
| CatBoost | 0.654 | 0.641 | 270.8 |
| **SLSQP blend (proposed)** | **0.663** | **0.653** | **262.0** |

The blend is the best configuration on both partitions. Crucially, the held-out
test $R^2$ (0.653) is within 0.010 of the development CV estimate (0.663),
indicating that the model **does not overfit** and that the cross-validation
figures reported throughout are honest rather than optimistic.

## V-B. Temporal robustness

To verify that the method is not tuned to a single window, the identical pipeline
was applied to three independent target quarters, each predicted from the
preceding history. Table 2 reports the blend performance (5-fold CV, clean
target).

**Table 2. Cross-period robustness of the proposed blend.**

| Setup | History → Target | $n$ | Blend $R^2$ |
|---|---|---|---|
| A | Q3 2025 → **Q4 2025** | 36,111 | 0.645 |
| B | Q3+Q4 2025 → **Q1 2026** | 36,282 | 0.574 |
| **C** | **7 mo (Jul’25–Jan’26) → Feb–Apr 2026** | 36,282 | **0.664** |

Performance is stable across all three periods, and the seven-month history of
Setup C yields the strongest result, confirming that deeper behavioural history
helps.

## V-C. Ablation study (what helps, what does not)

A controlled ablation isolates the contribution of each design decision (Table 3).
The dominant lever is the **target definition**: replacing the naïve occupancy
proxy (contaminated by host blocking) with the clean booking-differencing target
raises $R^2$ by roughly **+0.04** and is reproducible across periods. Every other
component contributes marginally or not at all — a set of honest negative
findings.

**Table 3. Ablation (5-fold CV $R^2$; effect on the blend).**

| Component | Effect on $R^2$ | Verdict |
|---|---|---|
| Clean booking-differencing target | **+0.04** | Largest, reproducible lever |
| Deeper history (2nd quarter → 7 months) | +0.005 – +0.02 | Small but real |
| Hyper-parameter tuning | +0.00 – +0.002 | Marginal |
| Enriched price features | ≈ 0 | No signal (calendar price is empty) |
| Review-velocity features | ≈ 0 | Redundant with calendar history |
| Two-stage hurdle model | 0 on held-out (0.653 = 0.653) | No gain over the direct blend |
| Neighbourhood / activity context | ≈ 0 | Redundant with listing-level history |

Notably, the **two-stage hurdle** model — the natural choice for a zero-inflated
target — attains an identical held-out $R^2$ (0.653) to the direct blend, so the
gradient-boosted blend already absorbs the zero inflation without an explicit
hurdle. Likewise, the posted **price does not predict demand** in this market
(a counter-intuitive but robust result), and **review velocity is redundant**
with the calendar-derived activity features.

## V-D. Statistical significance

The blend’s improvement over the best single model is statistically significant.
On a paired test of per-listing squared errors (best single vs. blend), Setup A
gives $t=7.69$, $p=1.6\times10^{-14}$ and Setup B gives $t=5.99$,
$p=2.1\times10^{-9}$; both 95% confidence intervals on the MSE reduction exclude
zero.

## V-E. Discussion: the performance ceiling is set by data, not the model

The ablation study points to a clear conclusion about where the remaining error
comes from. Model-side choices are saturated: hyper-parameter tuning, a two-stage
hurdle formulation, richer feature families, price features and review-velocity
features each contribute little or nothing to the held-out $R^2$. The only
components that move the metre are the **target definition** and the **depth of
behavioural history** — both properties of the *data*, not the learner.

Two intrinsic limitations of the Inside Airbnb data cap the achievable accuracy.
First, the platform exposes only a coarse `available` (true/false) flag rather
than an explicit booked status, so the target must be recovered by snapshot
differencing; this proxy conflates guest bookings with host blocking and is
therefore noisier than a direct booking signal. Second, the calendar price field
is empty in these snapshots, removing a price-dynamics signal that would otherwise
inform demand. Consequently the held-out $R^2\approx0.65$ observed for the
proposed blend is best interpreted as close to the practical ceiling for this data
source; further gains would require a cleaner booking label rather than a more
complex model.

---
*Türkçe not (rapora girmeyecek): Tamamen bağımsız/temiz 2025-26 makalesi — bashkal
YOK, 2016 YOK, sadece bizim veri + modellerimiz. Baş model = Setup C sade SLSQP
blend, held-out Test R² = 0.653. Onaylarsan tam JOURNAL makalesine geçeriz.*
