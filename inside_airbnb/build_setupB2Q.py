"""
build_setupB2Q.py
Setup B with TWO history quarters (mirrors competition Q1+Q2 -> Q3).
Recent quarter Q4 2025 features come from features_setupB; older quarter Q3 2025
features come from features_setupA (prefixed prev_); add quarter-over-quarter
deltas + half-year totals. Target Q1 2026, universe = January 2026 listings.
No new calendar reads -- reuses the two existing feature matrices.
Output: inside_airbnb/outputs/features_setupB2Q.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
b = pd.read_parquet(OUT/'features_setupB.parquet')           # Q4 behaviour + static + target
a = pd.read_parquet(OUT/'features_setupA.parquet')           # Q3 behaviour (older history)

# older-quarter (Q3 2025) behavioural columns from Setup A, prefixed prev_
prev_cols = {'q3_booked':'prev_booked','q3_occupied':'prev_occupied','q3_obs_days':'prev_obs_days',
             'q3_changes':'prev_changes','q3_booking_rate':'prev_booking_rate','q3_occ_rate':'prev_occ_rate',
             'booked_m07':'prev_booked_m07','booked_m08':'prev_booked_m08','booked_m09':'prev_booked_m09',
             'trend_sep_minus_jul':'prev_trend','lst_price_mean':'prev_price_mean',
             'lst_price_std':'prev_price_std','lst_price_cv':'prev_price_cv','lst_price_change':'prev_price_change'}
prev = a[['listing_id'] + list(prev_cols)].rename(columns=prev_cols)
X = b.merge(prev, on='listing_id', how='left')

# quarter-over-quarter (recent Q4 vs older Q3) + half-year aggregates
X['qoq_booked_change'] = X['q3_booked'] - X['prev_booked']
X['qoq_booking_rate_change'] = X['q3_booking_rate'] - X['prev_booking_rate']
X['qoq_occ_rate_change'] = X['q3_occ_rate'] - X['prev_occ_rate']
X['qoq_booked_ratio'] = X['q3_booked'] / (X['prev_booked'] + 1.0)
X['qoq_price_change'] = X['lst_price_mean'] / X['prev_price_mean'].replace(0, np.nan) - 1.0
X['h1_booked_total'] = X['q3_booked'].fillna(0) + X['prev_booked'].fillna(0)
X['h1_occ_rate'] = (X[['q3_occupied']].fillna(0).values.ravel() + a.set_index('listing_id')['q3_occupied'].reindex(X['listing_id']).fillna(0).values) / \
                   (X['q3_obs_days'].fillna(0) + X['prev_obs_days'].fillna(0) + 1.0)

X.to_parquet(OUT/'features_setupB2Q.parquet', index=False)
nn = X.select_dtypes(include='number').shape[1]; nc = X.select_dtypes(exclude='number').shape[1]
print(f'features_setupB2Q.parquet rows={len(X):,} cols={X.shape[1]} (num {nn}, cat {nc})')
print('has prev-quarter for', int(X['prev_booked'].notna().sum()), 'of', len(X), 'listings')
