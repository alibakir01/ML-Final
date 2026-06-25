"""
10_Residuals_and_Significance.py
--------------------------------
Post-hoc diagnostics requested in peer review:

  (1) Residual analysis for the zero-inflated, bounded target [0, 92]
      - residual vs. fitted, residual vs. true, residual histogram,
        and per-bin mean error to expose boundary behaviour.
  (2) A statistical-significance test for the XGBoost -> Blend MSE gap,
      so the improvement is not just attributed to CV variance:
      - paired t-test and Wilcoxon signed-rank on per-sample squared errors
        (XGBoost vs. SLSQP blend), with effect size and a 95% CI on the
        mean squared-error difference.

All numbers are recomputed from the saved out-of-fold (OOF) predictions on the
19,497-row development split, so they match Tables I and III of the paper.

Outputs:
  outputs/residual_analysis.png
  outputs/residuals_by_truebin.png
  outputs/significance_report.txt
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT = Path(__file__).resolve().parent.parent / 'outputs'

# ---- Load development target + OOF predictions --------------------------
y = pd.read_parquet(OUT / 'train_local.parquet')['NumReserveDays2016Q3'].astype(float).values

names = ['LinearRegression', 'RandomForest', 'GradientBoosting', 'XGBoost', 'MLP']
oofs = {n: np.clip(np.load(OUT / f'oof_{n}.npy'), 0, 92) for n in names}
M = np.column_stack([oofs[n] for n in names])
n = len(y)

# ---- Reproduce the SLSQP convex blend (paper Sec. IV.F) -----------------
def blend_mse(w):
    return float(mean_squared_error(y, np.clip(M @ w, 0, 92)))

k = M.shape[1]
res = minimize(blend_mse, np.full(k, 1.0 / k), method='SLSQP',
               bounds=[(0.0, 1.0)] * k,
               constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
               options={'maxiter': 500, 'ftol': 1e-9})
w = res.x.copy(); w[w < 1e-4] = 0.0; w = w / w.sum()
blend = np.clip(M @ w, 0, 92)

xgb = oofs['XGBoost']
print('Weights:', {nm: round(float(wi), 4) for nm, wi in zip(names, w)})
for nm in names:
    print(f'  {nm:<18s} MSE={mean_squared_error(y, oofs[nm]):8.3f}')
print(f'  {"Equal blend":<18s} MSE={mean_squared_error(y, np.clip(M.mean(1),0,92)):8.3f}')
print(f'  {"SLSQP blend":<18s} MSE={mean_squared_error(y, blend):8.3f}')

# =========================================================================
# (1) RESIDUAL ANALYSIS
# =========================================================================
res_xgb = y - xgb
res_bl = y - blend

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# (a) residual vs fitted (blend)
ax[0, 0].scatter(blend, res_bl, s=4, alpha=0.15, color='steelblue')
ax[0, 0].axhline(0, color='k', lw=1)
ax[0, 0].set_xlabel('Predicted booked nights (blend)')
ax[0, 0].set_ylabel('Residual  (y - y_hat)')
ax[0, 0].set_title('(a) Residual vs. fitted')

# (b) residual vs true, with boundary bands highlighted
ax[0, 1].scatter(y, res_bl, s=4, alpha=0.15, color='darkorange')
ax[0, 1].axhline(0, color='k', lw=1)
ax[0, 1].axvspan(-0.5, 2.5, color='red', alpha=0.08, label='zero-inflated edge [0,2]')
ax[0, 1].axvspan(89.5, 92.5, color='green', alpha=0.08, label='upper bound [90,92]')
ax[0, 1].set_xlabel('True booked nights  y')
ax[0, 1].set_ylabel('Residual  (y - y_hat)')
ax[0, 1].set_title('(b) Residual vs. true target')
ax[0, 1].legend(fontsize=8, loc='lower left')

# (c) residual histogram
ax[1, 0].hist(res_bl, bins=60, color='slategray', edgecolor='white')
ax[1, 0].axvline(0, color='red', lw=1)
ax[1, 0].set_xlabel('Residual (blend)')
ax[1, 0].set_ylabel('Count')
ax[1, 0].set_title(f'(c) Residual distribution  (mean={res_bl.mean():.2f}, sd={res_bl.std():.2f})')

# (d) Q-Q plot of residuals
stats.probplot(res_bl, dist='norm', plot=ax[1, 1])
ax[1, 1].set_title('(d) Normal Q-Q of residuals')

plt.tight_layout()
plt.savefig(OUT / 'residual_analysis.png', dpi=130)
plt.close()

# ---- Mean signed error per true-value bin (boundary diagnosis) ----------
bins = [(-0.5, 0.5), (0.5, 5.5), (5.5, 20.5), (20.5, 50.5),
        (50.5, 80.5), (80.5, 89.5), (89.5, 92.5)]
labels = ['0', '1-5', '6-20', '21-50', '51-80', '81-89', '90-92']
mean_err, counts = [], []
for lo, hi in bins:
    m = (y > lo) & (y <= hi)
    counts.append(int(m.sum()))
    mean_err.append(float(res_bl[m].mean()) if m.any() else 0.0)

fig, ax1 = plt.subplots(figsize=(9, 5))
colmap = ['firebrick' if e < 0 else 'seagreen' for e in mean_err]
ax1.bar(labels, mean_err, color=colmap, alpha=0.85)
ax1.axhline(0, color='k', lw=1)
ax1.set_xlabel('True booked-nights bin')
ax1.set_ylabel('Mean signed residual  (y - y_hat)')
ax1.set_title('Systematic bias by target bin — blend on dev OOF')
for i, (e, c) in enumerate(zip(mean_err, counts)):
    ax1.text(i, e, f'{e:+.1f}\n(n={c})', ha='center',
             va='bottom' if e >= 0 else 'top', fontsize=8)
plt.tight_layout()
plt.savefig(OUT / 'residuals_by_truebin.png', dpi=130)
plt.close()

# =========================================================================
# (2) SIGNIFICANCE TEST:  XGBoost  vs.  SLSQP blend
# =========================================================================
se_xgb = res_xgb ** 2
se_bl = res_bl ** 2
d = se_xgb - se_bl                       # positive => blend better on that row
mean_d = d.mean()
sd_d = d.std(ddof=1)
se_mean = sd_d / np.sqrt(n)

t_stat, p_t = stats.ttest_rel(se_xgb, se_bl)
w_stat, p_w = stats.wilcoxon(se_xgb, se_bl, zero_method='wilcox')
ci_lo, ci_hi = mean_d - 1.96 * se_mean, mean_d + 1.96 * se_mean
cohen_dz = mean_d / sd_d

lines = []
lines.append('SIGNIFICANCE OF THE XGBoost -> SLSQP-BLEND MSE IMPROVEMENT')
lines.append('=' * 60)
lines.append(f'n (paired dev OOF samples)      : {n}')
lines.append(f'MSE XGBoost                     : {se_xgb.mean():.4f}')
lines.append(f'MSE SLSQP blend                 : {se_bl.mean():.4f}')
lines.append(f'Mean SE difference (XGB - blend): {mean_d:.4f}')
lines.append(f'95% CI on mean SE difference    : [{ci_lo:.4f}, {ci_hi:.4f}]')
lines.append(f'Paired t-test                   : t={t_stat:.4f}, p={p_t:.3e}')
lines.append(f'Wilcoxon signed-rank            : W={w_stat:.1f}, p={p_w:.3e}')
lines.append(f"Effect size (Cohen's d_z)       : {cohen_dz:.4f}")
verdict = ('SIGNIFICANT (p<0.05): the blend genuinely lowers squared error; '
           'CI excludes 0.') if (p_t < 0.05 and ci_lo > 0) else \
          ('NOT significant at 0.05: gap is within CV noise.')
lines.append(f'Verdict                         : {verdict}')
report = '\n'.join(lines)
print('\n' + report)
(OUT / 'significance_report.txt').write_text(report, encoding='utf-8')

json.dump({nm: float(wi) for nm, wi in zip(names, w)},
          open(OUT / 'blend_weights_recomputed.json', 'w'), indent=2)
print('\nSaved: residual_analysis.png, residuals_by_truebin.png, significance_report.txt')
