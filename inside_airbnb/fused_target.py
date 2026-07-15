"""
fused_target.py -- #1: review-fused Feb-Mar-Apr 2026 target.
Review-based occupancy estimate (San Francisco model):
  est_nights = reviews_in_window * (1/review_rate) * avg_stay,  review_rate=0.5, avg_stay=3
Reviews counted from the April-2026 snapshot (covers Feb-Apr stays).
Fusion is ADDITIVE only: y_fused = max(y_diff_clean, est_nights)  -- reviews can
confirm extra bookings the calendar missed but never reduce the count.
Output: target_setupC_fused.csv (y_fused) + stats.
"""
from pathlib import Path
import numpy as np, pandas as pd
P26 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2026_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
LO, HI = pd.Timestamp('2026-02-01'), pd.Timestamp('2026-05-01')

r = pd.read_csv(P26/'April2026'/'reviews.csv', usecols=['listing_id', 'date'])
r['date'] = pd.to_datetime(r['date'], errors='coerce')
r = r[(r['date'] >= LO) & (r['date'] < HI)]
rc = r.groupby('listing_id').size().rename('rev_febapr')
est = (rc * (1/0.5) * 3).clip(0, 89).rename('y_review')   # SF occupancy estimate

d = pd.read_csv(OUT/'target_setupC_clean.csv')             # listing_id, y_clean, y_orig
d = d.merge(est, on='listing_id', how='left')
d['y_review'] = d['y_review'].fillna(0)
d['y_fused'] = np.maximum(d['y_clean'], d['y_review']).astype('int32')
d[['listing_id', 'y_fused', 'y_clean', 'y_review']].to_csv(OUT/'target_setupC_fused.csv', index=False)

def st(n, v): return f'{n:<9} mean={v.mean():.2f} median={np.median(v):.1f} std={v.std():.2f} %zero={np.mean(v==0)*100:.1f}'
rep = ('REVIEW-FUSED TARGET (Setup C, Feb-Mar-Apr 2026)\n' + '='*50 + '\n' +
       st('y_clean', d['y_clean'].values) + '\n' + st('y_review', d['y_review'].values) + '\n' +
       st('y_fused', d['y_fused'].values) + '\n' +
       f'listings where review added nights: {(d["y_fused"] > d["y_clean"]).sum():,} ({(d["y_fused"]>d["y_clean"]).mean()*100:.1f}%)' + '\n' +
       f'corr(y_fused, y_clean) = {np.corrcoef(d["y_fused"], d["y_clean"])[0,1]:.3f}')
print(rep); (OUT/'target_setupC_fused_stats.txt').write_text(rep, encoding='utf-8')
