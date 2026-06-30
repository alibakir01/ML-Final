"""
add_price_features.py -- enrich Setup B-2Q with a 6-month price trajectory
(Jul..Dec 2025) from the monthly listings snapshots (calendar price is empty).
Output: features_setupB2Q_price.parquet  (adds p6_* price features)
"""
from pathlib import Path
import numpy as np, pandas as pd
B = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
MONTHS = [(B/'Q3'/'July2025', '07'), (B/'Q3'/'August2025', '08'), (B/'Q3'/'September2025', '09'),
          (B/'Q4'/'October2025', '10'), (B/'Q4'/'November2025', '11'), (B/'Q4'/'December2025', '12')]
money = lambda s: pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

parts = []
for folder, m in MONTHS:
    d = pd.read_csv(folder/'listings.csv', usecols=['id', 'price'], low_memory=False)
    d['price'] = money(d['price'])
    parts.append(d.rename(columns={'id': 'listing_id', 'price': f'p_{m}'}).set_index('listing_id'))
pr = pd.concat(parts, axis=1)
pc = [c for c in pr.columns if c.startswith('p_')]
pr['p6_mean'] = pr[pc].mean(axis=1)
pr['p6_std'] = pr[pc].std(axis=1)
pr['p6_cv'] = pr['p6_std'] / pr['p6_mean'].replace(0, np.nan)
pr['p6_min'] = pr[pc].min(axis=1)
pr['p6_max'] = pr[pc].max(axis=1)
pr['p6_range'] = pr['p6_max'] - pr['p6_min']
pr['p6_last'] = pr['p_12']
pr['p6_pct_change'] = (pr['p_12'] - pr['p_07']) / pr['p_07'].replace(0, np.nan)
pr['p6_q4_vs_q3'] = pr[['p_10', 'p_11', 'p_12']].mean(axis=1) / pr[['p_07', 'p_08', 'p_09']].mean(axis=1).replace(0, np.nan) - 1.0
pr['p6_n_distinct'] = pr[pc].nunique(axis=1)
keepcols = ['p6_mean','p6_std','p6_cv','p6_min','p6_max','p6_range','p6_last','p6_pct_change','p6_q4_vs_q3','p6_n_distinct']
pr = pr[keepcols].reset_index()

X = pd.read_parquet(OUT/'features_setupB2Q.parquet').merge(pr, on='listing_id', how='left')
X.to_parquet(OUT/'features_setupB2Q_price.parquet', index=False)
print('features_setupB2Q_price.parquet cols=%d (added %d price features)' % (X.shape[1], len(keepcols)))
print('price coverage:', int(X['p6_mean'].notna().sum()), 'of', len(X))
