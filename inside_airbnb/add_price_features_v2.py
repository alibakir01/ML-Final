"""
add_price_features_v2.py -- fair retry of the price family using months that
ACTUALLY contain price (Jul-Nov 2025, ~59% each; Dec 2025 is empty so excluded).
Adds a 5-month trajectory, a missing-price flag, and a neighbourhood-relative
price (more predictive than the raw level). All <= Nov 2025 -> leakage-safe.
Replaces the old p6_* columns. Output: features_setupB2Q_final2.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
B = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
MONTHS = [(B/'Q3'/'July2025', '07'), (B/'Q3'/'August2025', '08'), (B/'Q3'/'September2025', '09'),
          (B/'Q4'/'October2025', '10'), (B/'Q4'/'November2025', '11')]   # filled months only
money = lambda s: pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

parts = []
for folder, m in MONTHS:
    d = pd.read_csv(folder/'listings.csv', usecols=['id', 'price'], low_memory=False)
    d['price'] = money(d['price'])
    parts.append(d.set_index('id')['price'].rename(f'p_{m}'))
pr = pd.concat(parts, axis=1)
# neighbourhood from November snapshot (robust: detect available column)
nov = pd.read_csv(B/'Q4'/'November2025'/'listings.csv', low_memory=False)
nbcol = next((c for c in ['neighbourhood_cleansed', 'neighbourhood', 'neighbourhood_group_cleansed'] if c in nov.columns), None)
nbhd = nov.set_index('id')[nbcol] if nbcol else pd.Series('na', index=pr.index)
pc = list(pr.columns)
pr['p5_mean'] = pr[pc].mean(axis=1)
pr['p5_std'] = pr[pc].std(axis=1)
pr['p5_cv'] = pr['p5_std'] / pr['p5_mean'].replace(0, np.nan)
pr['p5_min'] = pr[pc].min(axis=1)
pr['p5_max'] = pr[pc].max(axis=1)
pr['p5_last'] = pr['p_11']
pr['p5_trend'] = (pr['p_11'] - pr['p_07']) / pr['p_07'].replace(0, np.nan)
pr['price_missing'] = pr['p5_mean'].isna().astype('int8')
# neighbourhood-relative price (price level vs same-neighbourhood median)
nb = pd.DataFrame({'price': pr['p5_mean'], 'nb': nbhd.reindex(pr.index)})
med = nb.groupby('nb')['price'].transform('median')
pr['price_rel_nbhd'] = pr['p5_mean'] / med.replace(0, np.nan)
keep = ['p5_mean','p5_std','p5_cv','p5_min','p5_max','p5_last','p5_trend','price_missing','price_rel_nbhd']
pr = pr[keep].reset_index().rename(columns={'id': 'listing_id'})

X = pd.read_parquet(OUT/'features_setupB2Q_final.parquet')
X = X[[c for c in X.columns if not c.startswith('p6_')]]   # drop old (Dec-diluted) price
X = X.merge(pr, on='listing_id', how='left')
X.to_parquet(OUT/'features_setupB2Q_final2.parquet', index=False)
print('features_setupB2Q_final2.parquet cols=%d' % X.shape[1])
print('price coverage (p5_mean): %.1f%%' % (X['p5_mean'].notna().mean()*100))
