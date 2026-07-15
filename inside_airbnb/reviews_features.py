"""
reviews_features.py -- review-velocity features for Setup A (cutoff 2025-10-01).
Reviews are strong, leakage-safe demand proxies (all dated <= Aug 2025).
Output: reviews_setupA.parquet (listing_id + review features)
"""
from pathlib import Path
import numpy as np, pandas as pd
SEP = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb\Q3\September2025')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
REF = pd.Timestamp('2025-10-01')

r = pd.read_csv(SEP/'reviews.csv', usecols=['listing_id', 'date'])
r['date'] = pd.to_datetime(r['date'], errors='coerce')
r = r[r['date'] < REF]
age = (REF - r['date']).dt.days
r = r.assign(age=age)

g = r.groupby('listing_id')
feat = pd.DataFrame({
    'rev_total': g.size(),
    'rev_30': g['age'].apply(lambda a: (a < 30).sum()),
    'rev_90': g['age'].apply(lambda a: (a < 90).sum()),
    'rev_180': g['age'].apply(lambda a: (a < 180).sum()),
    'rev_365': g['age'].apply(lambda a: (a < 365).sum()),
    'rev_prev90': g['age'].apply(lambda a: ((a >= 90) & (a < 180)).sum()),
    'rev_recency_days': g['age'].min(),
})
feat['rev_per_month_recent'] = feat['rev_90'] / 3.0
feat['rev_accel'] = feat['rev_90'] - feat['rev_prev90']
feat['rev_velocity_365'] = feat['rev_365'] / 12.0
feat = feat.reset_index().rename(columns={'index': 'listing_id'})
feat.to_parquet(OUT/'reviews_setupA.parquet', index=False)
print('reviews_setupA.parquet: listings with reviews =', len(feat))
print(feat[['rev_total', 'rev_90', 'rev_365', 'rev_recency_days']].describe().round(1).to_string())
