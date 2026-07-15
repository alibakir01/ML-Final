"""
analyze_setup.py --tag <setupA|setupB>
Significance test (best single model vs SLSQP blend, paired on per-sample squared
errors) + residual analysis on the [0,92] boundary. Uses oof_<tag>.npz.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error

ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
d = np.load(OUT / f'oof_{a.tag}.npz', allow_pickle=True)
y = d['y']; names = list(d['names']); blend = d['blend']
mses = {n: mean_squared_error(y, d[n]) for n in names}
best = min(mses, key=mses.get); xb = d[best]

# significance: per-sample squared-error difference (best single - blend)
db = (y - xb) ** 2 - (y - blend) ** 2
n = len(y); mean_d = db.mean(); se = db.std(ddof=1) / np.sqrt(n)
t, p = stats.ttest_rel((y - xb) ** 2, (y - blend) ** 2)
w, pw = stats.wilcoxon((y - xb) ** 2, (y - blend) ** 2, zero_method='wilcox')
ci = (mean_d - 1.96 * se, mean_d + 1.96 * se)
rep = [f'SIGNIFICANCE [{a.tag}] best single ({best}, MSE={mses[best]:.2f}) vs SLSQP blend (MSE={mean_squared_error(y, blend):.2f})',
       '=' * 66,
       f'n                       : {n}',
       f'mean SE diff (best-blend): {mean_d:.4f}',
       f'95% CI                  : [{ci[0]:.4f}, {ci[1]:.4f}]',
       f'paired t-test           : t={t:.4f}, p={p:.3e}',
       f'Wilcoxon                : W={w:.1f}, p={pw:.3e}',
       f"Cohen's d_z             : {mean_d / db.std(ddof=1):.4f}",
       'verdict                 : ' + ('SIGNIFICANT (p<0.05, CI excludes 0)' if (p < 0.05 and ci[0] > 0) else 'not significant')]
txt = '\n'.join(rep); print(txt)
(OUT / f'significance_{a.tag}.txt').write_text(txt, encoding='utf-8')

# residual analysis
res = y - blend
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0, 0].scatter(blend, res, s=4, alpha=0.12, color='steelblue'); ax[0, 0].axhline(0, color='k', lw=1)
ax[0, 0].set_xlabel('Predicted (blend)'); ax[0, 0].set_ylabel('Residual'); ax[0, 0].set_title('(a) Residual vs. fitted')
ax[0, 1].scatter(y, res, s=4, alpha=0.12, color='darkorange'); ax[0, 1].axhline(0, color='k', lw=1)
ax[0, 1].axvspan(-.5, 2.5, color='red', alpha=.08); ax[0, 1].axvspan(89.5, 92.5, color='green', alpha=.08)
ax[0, 1].set_xlabel('True y'); ax[0, 1].set_ylabel('Residual'); ax[0, 1].set_title('(b) Residual vs. true')
ax[1, 0].hist(res, bins=60, color='slategray', edgecolor='white'); ax[1, 0].axvline(0, color='red', lw=1)
ax[1, 0].set_xlabel('Residual'); ax[1, 0].set_title(f'(c) Residuals (mean={res.mean():.2f}, sd={res.std():.2f})')
stats.probplot(res, dist='norm', plot=ax[1, 1]); ax[1, 1].set_title('(d) Normal Q-Q')
fig.tight_layout(); fig.savefig(OUT / f'residuals_{a.tag}.png', dpi=150); plt.close(fig)

bins = [(-.5, .5), (.5, 5.5), (5.5, 20.5), (20.5, 50.5), (50.5, 80.5), (80.5, 89.5), (89.5, 92.5)]
labs = ['0', '1-5', '6-20', '21-50', '51-80', '81-89', '90-92']; me = []; cs = []
for lo, hi in bins:
    m = (y > lo) & (y <= hi); cs.append(int(m.sum())); me.append(float(res[m].mean()) if m.any() else 0)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labs, me, color=['firebrick' if e < 0 else 'seagreen' for e in me], alpha=.85); ax.axhline(0, color='k', lw=1)
ax.set_xlabel('True booked-nights bin'); ax.set_ylabel('Mean signed residual'); ax.set_title(f'Bias by bin [{a.tag}]')
for i, (e, c) in enumerate(zip(me, cs)): ax.text(i, e, f'{e:+.1f}\n(n={c})', ha='center', va='bottom' if e >= 0 else 'top', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / f'residual_bins_{a.tag}.png', dpi=150); plt.close(fig)
print(f'saved significance_{a.tag}.txt, residuals_{a.tag}.png, residual_bins_{a.tag}.png')
