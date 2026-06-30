"""
enhance_v3.py -- add #2 (activity/segmentation) + #3 (neighbourhood demand context)
features to Setup C v2. All from leakage-safe history features (no target use).
Output: features_setupC_v3.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
X = pd.read_parquet(OUT/'features_setupC_v2.parquet')

# ---- #2 activity / segmentation (history-based) ----
X['is_active'] = (X['h7_booked_total'].fillna(0) > 0).astype('int8')
X['recent_active'] = (X['traj_last'].fillna(0) > 0).astype('int8')   # traj_last = Jan booked
X['inactive_dead'] = ((X['h7_booked_total'].fillna(0) == 0) &
                      (X['rev_recency_days'].fillna(9999) > 180)).astype('int8')
X['active_intensity'] = X['hjan_booking_rate'].fillna(0) * X['is_active']

# ---- #3 neighbourhood demand context (group means on history; leakage-safe) ----
def add_group_ctx(df, key, prefix):
    if key not in df.columns:
        return df
    g = df.groupby(key)
    df[f'{prefix}_mean_rate'] = g['hjan_booking_rate'].transform('mean')
    df[f'{prefix}_mean_occ'] = g['hq4_occ_rate'].transform('mean')
    df[f'{prefix}_density'] = g['listing_id'].transform('count')
    df[f'{prefix}_active_share'] = g['is_active'].transform('mean')
    df[f'rate_vs_{prefix}'] = df['hjan_booking_rate'].fillna(0) - df[f'{prefix}_mean_rate']
    return df

X = add_group_ctx(X, 'neighbourhood_cleansed', 'nb')
X = add_group_ctx(X, 'geo_cluster', 'geo')

X.to_parquet(OUT/'features_setupC_v3.parquet', index=False)
print('features_setupC_v3.parquet cols=%d (added activity + neighbourhood context)' % X.shape[1])
print('active listings: %.1f%% | dead-inactive: %.1f%%' %
      (X['is_active'].mean()*100, X['inactive_dead'].mean()*100))
