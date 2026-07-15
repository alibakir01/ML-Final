"""
eda_tour.py -- deep EDA on Setup A (clean target) + review features.
Outputs a multi-panel figure and a text summary to guide feature/target work.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
X = pd.read_parquet(OUT/'features_setupA_clean.parquet')
rev = pd.read_parquet(OUT/'reviews_setupA.parquet')
X = X.merge(rev, on='listing_id', how='left')
y = X['y_clean']
num = X.select_dtypes(include='number').drop(columns=['listing_id', 'y_clean'])

# correlations with target
corr = num.corrwith(y).dropna().sort_values(key=np.abs, ascending=False)
top = corr.head(20)[::-1]

# cold-start vs warm
warm = X['q3_obs_days'].notna()
has_rev = X['rev_total'].notna() & (X['rev_total'] > 0)
lines = ['DEEP EDA -- Setup A (clean target y_clean, n=%d)' % len(X), '='*60,
         f'target: mean={y.mean():.2f} median={np.median(y):.1f} std={y.std():.2f} %zero={np.mean(y==0)*100:.1f}',
         f'warm listings (Q3 history): {warm.sum():,} ({warm.mean()*100:.1f}%)  | cold-start: {(~warm).sum():,}',
         f'  warm target mean={y[warm].mean():.2f}  | cold-start target mean={y[~warm].mean():.2f}',
         f'listings with any review: {has_rev.sum():,} ({has_rev.mean()*100:.1f}%)',
         f'  with-review target mean={y[has_rev].mean():.2f}  | no-review target mean={y[~has_rev].mean():.2f}',
         '', 'TOP-15 |correlation| with target:']
for k, v in corr.head(15).items():
    lines.append(f'  {k:<26} {v:+.3f}')
rep = '\n'.join(lines)
print(rep); (OUT/'eda_tour_summary.txt').write_text(rep, encoding='utf-8')

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
ax[0, 0].barh(range(len(top)), top.values, color=['firebrick' if v < 0 else 'steelblue' for v in top.values])
ax[0, 0].set_yticks(range(len(top))); ax[0, 0].set_yticklabels(top.index, fontsize=8)
ax[0, 0].set_title('Top-20 features by |correlation| with target'); ax[0, 0].set_xlabel('Pearson r')

ax[0, 1].hist(y, bins=93, color='seagreen'); ax[0, 1].set_title('Target (clean) distribution'); ax[0, 1].set_xlabel('y_clean')

# review velocity vs target (binned)
rb = pd.cut(X['rev_90'].fillna(0), bins=[-1, 0, 1, 3, 10, 1000], labels=['0', '1', '2-3', '4-10', '10+'])
mb = y.groupby(rb, observed=True).mean()
ax[1, 0].bar(mb.index.astype(str), mb.values, color='darkorange')
ax[1, 0].set_title('Mean target by recent review count (rev_90)'); ax[1, 0].set_xlabel('reviews in last 90d')
ax[1, 0].set_ylabel('mean booked nights')

# occ_rate (Q3) vs target scatter
ax[1, 1].scatter(X['q3_occ_rate'], y, s=4, alpha=0.1, color='steelblue')
ax[1, 1].set_title('Q3 occupancy rate vs Q4 target'); ax[1, 1].set_xlabel('q3_occ_rate'); ax[1, 1].set_ylabel('y_clean')
fig.tight_layout(); fig.savefig(OUT/'eda_tour.png', dpi=140); plt.close(fig)
print('\nsaved eda_tour.png, eda_tour_summary.txt')
