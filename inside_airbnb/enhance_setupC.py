"""
enhance_setupC.py -- add fine-grained 7-month booking-trajectory (recency) features
to Setup C: monthly booked counts Jul'25..Jan'26 -> slope, acceleration,
recent-vs-older, volatility. Output: features_setupC_v2.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
X = pd.read_parquet(OUT/'features_setupC.parquet')
a = pd.read_parquet(OUT/'features_setupA.parquet')[['listing_id', 'booked_m07', 'booked_m08', 'booked_m09']]
b = pd.read_parquet(OUT/'features_setupB.parquet')[['listing_id', 'booked_m10', 'booked_m11', 'booked_m12']]
X = X.merge(a, on='listing_id', how='left').merge(b, on='listing_id', how='left')

mcols = ['booked_m07', 'booked_m08', 'booked_m09', 'booked_m10', 'booked_m11', 'booked_m12', 'hjan_booked']
M = X[mcols].fillna(0).to_numpy(dtype=float)            # 7-month monthly booked trajectory
t = np.arange(7); tc = t - t.mean()
# linear slope per row: cov(t,y)/var(t)
X['traj_slope'] = (M * tc).sum(1) / (tc**2).sum()
X['traj_recent3'] = M[:, 4:].mean(1)                    # Nov,Dec,Jan
X['traj_older3'] = M[:, :3].mean(1)                     # Jul,Aug,Sep
X['traj_recent_older_ratio'] = X['traj_recent3'] / (X['traj_older3'] + 1.0)
X['traj_accel'] = M[:, 6] - M[:, 5]                     # Jan - Dec
X['traj_std'] = M.std(1)
X['traj_max'] = M.max(1)
X['traj_nonzero_months'] = (M > 0).sum(1)
X['traj_last'] = M[:, 6]                                # January
X = X.drop(columns=mcols)                               # keep engineered, drop raw monthly (in other blocks already)
X.to_parquet(OUT/'features_setupC_v2.parquet', index=False)
print('features_setupC_v2.parquet cols=%d (added 9 trajectory features)' % X.shape[1])
