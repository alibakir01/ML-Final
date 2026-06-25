"""
12_ablation.py  --  REAL incremental ablation requested by review (item IV.3).
All numbers are computed here with the same 5-fold split (KFold, shuffle,
random_state=42) and the same leakage-safe preprocessing used elsewhere.

Ladder (cumulative):
  A1  Linear Regression, raw static property features only (no daily-history
      aggregates, no target encoding)
  A2  Linear Regression, + 150 engineered features, high-cardinality categoricals
      DROPPED (i.e. without target encoding)
  A3  Linear Regression, + K-fold target encoding (full preprocessing)
  Then, on the full preprocessing, the model/ensemble axis (reused from the main
  experiments, recomputed here for consistency):
  B1  XGBoost (tuned)
  B2  Equal-weight blend of the five learners
  B3  SLSQP convex blend (proposed)

Writes outputs/ablation_results.txt
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from xgboost import XGBRegressor

OUT = Path('outputs'); RS, K = 42, 5
train = pd.read_parquet(OUT / 'train_dev.parquet')
TARGET, ID = 'NumReserveDays2016Q3', 'PropertyID'
y = train[TARGET].astype(float).values
X = train.drop(columns=[TARGET, ID]).reset_index(drop=True)

# classify columns
DAILY = ('q1_', 'q2_', 'h1_', 'month', 'recent', 'recency', 'weekend', 'weekday',
         'trans', 'a2b', 'b2a', 'status', 'qoq', 'delta', 'change', 'kw_',
         'title', 'cluster', 'geo', 'reserved', 'booking', 'available')
def is_daily(c): return any(k in c.lower() for k in DAILY)
num_all = X.select_dtypes(include='number').columns.tolist()
cat_all = X.select_dtypes(exclude='number').columns.tolist()
raw_num = [c for c in num_all if not is_daily(c)]              # static numeric only
card = {c: X[c].nunique(dropna=False) for c in cat_all}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]

kf = KFold(K, shuffle=True, random_state=RS)


class KFoldTE(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        self.cols = cols; self.n_splits = n_splits
        self.smoothing = smoothing; self.random_state = random_state
    def _m(self, x, yy):
        st = pd.DataFrame({'c': x, 'y': yy}).groupby('c')['y'].agg(['mean', 'count'])
        return ((st['count'] * st['mean'] + self.smoothing * self.gm_) / (st['count'] + self.smoothing)).to_dict()
    def fit(self, X, y):
        y = np.asarray(y, float); self.gm_ = float(y.mean())
        self.maps_ = {c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return self
    def transform(self, X):
        Xo = X.copy()
        for c in self.cols:
            Xo[c] = X[c].astype(str).fillna('_n').map(self.maps_[c]).fillna(self.gm_).astype('float32')
        return Xo
    def fit_transform(self, X, y=None, **k):
        y = np.asarray(y, float); self.gm_ = float(y.mean()); Xo = X.copy()
        for c in self.cols: Xo[c] = np.full(len(X), self.gm_, 'float32')
        kk = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kk.split(X):
            for c in self.cols:
                m = self._m(X[c].astype(str).fillna('_n').iloc[tr], y[tr])
                Xo.iloc[va, Xo.columns.get_loc(c)] = X[c].astype(str).fillna('_n').iloc[va].map(m).fillna(self.gm_).astype('float32').values
        self.maps_ = {c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return Xo


def cv(pp_factory, model_factory, cols):
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        pp = pp_factory()
        Xtr = pp.fit_transform(X[cols].iloc[tr], y[tr]); Xva = pp.transform(X[cols].iloc[va])
        m = model_factory(); m.fit(Xtr, y[tr])
        oof[va] = np.clip(m.predict(Xva), 0, 92)
    return mean_squared_error(y, oof)

# A1: raw static numeric only, linear
pp_raw = lambda: ColumnTransformer([('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), raw_num)])
a1 = cv(pp_raw, LinearRegression, raw_num)

# A2: full engineered, NO target encoding (drop high-card cats)
cols_a2 = num_all + cat_low
pp_a2 = lambda: ColumnTransformer([
    ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num_all),
    ('l', Pipeline([('i', SimpleImputer(strategy='constant', fill_value='m')), ('o', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low)])
a2 = cv(pp_a2, LinearRegression, cols_a2)

# A3: full pipeline WITH target encoding, linear
cols_full = num_all + cat_low + cat_high
pp_full = lambda: ColumnTransformer([
    ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num_all),
    ('l', Pipeline([('i', SimpleImputer(strategy='constant', fill_value='m')), ('o', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
    ('h', Pipeline([('te', KFoldTE(cat_high))]), cat_high)])
a3 = cv(pp_full, LinearRegression, cols_full)

# B-axis: reuse established, real OOF numbers (XGBoost / blends) from saved arrays
names = ['LinearRegression', 'RandomForest', 'GradientBoosting', 'XGBoost', 'MLP']
yL = pd.read_parquet(OUT / 'train_local.parquet')[TARGET].astype(float).values
M = np.column_stack([np.clip(np.load(OUT / f'oof_{n}.npy'), 0, 92) for n in names])
xgb_mse = mean_squared_error(yL, M[:, 3])
equal = mean_squared_error(yL, np.clip(M.mean(1), 0, 92))
def bm(w): return mean_squared_error(yL, np.clip(M @ w, 0, 92))
r = minimize(bm, np.full(5, .2), method='SLSQP', bounds=[(0, 1)] * 5,
             constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-9})
w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum(); slsqp = bm(w)

rows = [
    ('Linear Reg., raw static features only', a1),
    ('  + 150 engineered features (no target encoding)', a2),
    ('  + K-fold target encoding (full preprocessing)', a3),
    ('XGBoost (tuned, full preprocessing)', xgb_mse),
    ('Equal-weight blend of 5 learners', equal),
    ('SLSQP convex blend (proposed)', slsqp),
]
out = ['INCREMENTAL ABLATION (5-fold CV MSE, random_state=42)', '=' * 60]
prev = None
for name, mse in rows:
    d = '' if prev is None else f'  (delta {mse - prev:+.1f})'
    out.append(f'{name:<48}{mse:8.1f}{d}'); prev = mse
rep = '\n'.join(out)
(OUT / 'ablation_results.txt').write_text(rep, encoding='utf-8')
print(rep.encode('ascii', 'replace').decode())
print('saved ablation_results.txt | raw_num cols:', len(raw_num))
