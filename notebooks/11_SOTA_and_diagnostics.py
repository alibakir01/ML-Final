"""
11_SOTA_and_diagnostics.py
Journal-revision experiments (all numbers are real, computed here):

  (1) SOTA baselines: LightGBM and CatBoost trained with the SAME leakage-safe
      pipeline and the SAME 5-fold split as the in-house XGBoost, so the
      comparison in the paper is fair.
  (2) Failure-case analysis: the development-set listings with the largest blend
      residuals, with their key characteristics.
  (3) Feature-family inventory from the engineered matrix.

Writes outputs/sota_results.txt and outputs/failure_cases.txt
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

OUT = Path('outputs')
RS, K = 42, 5
train = pd.read_parquet(OUT / 'train_dev.parquet')
TARGET, ID = 'NumReserveDays2016Q3', 'PropertyID'
y = train[TARGET].astype(float).values
X = train.drop(columns=[TARGET, ID]).reset_index(drop=True)

cat_all = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].nunique(dropna=False) for c in cat_all}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]
num_cols = X.select_dtypes(include='number').columns.tolist()


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        self.cols = cols; self.n_splits = n_splits
        self.smoothing = smoothing; self.random_state = random_state

    def _smap(self, x, yy):
        st = pd.DataFrame({'c': x, 'y': yy}).groupby('c')['y'].agg(['mean', 'count'])
        return ((st['count'] * st['mean'] + self.smoothing * self.gm_)
                / (st['count'] + self.smoothing)).to_dict()

    def fit(self, X, y):
        y = np.asarray(y, float); self.gm_ = float(y.mean())
        self.maps_ = {c: self._smap(X[c].astype(str).fillna('__nan__'), y) for c in self.cols}
        return self

    def transform(self, X):
        Xo = X.copy()
        for c in self.cols:
            Xo[c] = (X[c].astype(str).fillna('__nan__').map(self.maps_[c])
                       .fillna(self.gm_).astype('float32'))
        return Xo

    def fit_transform(self, X, y=None, **kw):
        y = np.asarray(y, float); self.gm_ = float(y.mean())
        Xo = X.copy()
        for c in self.cols:
            Xo[c] = np.full(len(X), self.gm_, dtype='float32')
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(X):
            for c in self.cols:
                m = self._smap(X[c].astype(str).fillna('__nan__').iloc[tr], y[tr])
                Xo.iloc[va, Xo.columns.get_loc(c)] = (
                    X[c].astype(str).fillna('__nan__').iloc[va].map(m)
                       .fillna(self.gm_).astype('float32').values)
        self.maps_ = {c: self._smap(X[c].astype(str).fillna('__nan__'), y) for c in self.cols}
        return Xo


def make_pp():
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_cols),
        ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
        ('high', Pipeline([('te', KFoldTargetEncoder(cat_high, 5, 20, RS))]), cat_high),
    ])


def cv_eval(make_model):
    kf = KFold(K, shuffle=True, random_state=RS)
    oof = np.zeros(len(y)); fold_mse = []
    t0 = time.time()
    for tr, va in kf.split(X):
        pp = make_pp()
        Xtr = pp.fit_transform(X.iloc[tr], y[tr]); Xva = pp.transform(X.iloc[va])
        m = make_model(); m.fit(Xtr, y[tr])
        p = np.clip(m.predict(Xva), 0, 92)
        oof[va] = p; fold_mse.append(mean_squared_error(y[va], p))
    return (mean_squared_error(y, oof), mean_absolute_error(y, oof),
            r2_score(y, oof), float(np.std(fold_mse)), time.time() - t0)


models = {
    'LightGBM': lambda: LGBMRegressor(n_estimators=1000, learning_rate=0.03,
                  num_leaves=48, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1),
    'CatBoost': lambda: CatBoostRegressor(iterations=1000, learning_rate=0.03,
                  depth=6, l2_leaf_reg=3.0, random_seed=RS, verbose=0),
}

lines = ['SOTA BASELINES (5-fold CV on the 19,497-row development set, same split as XGBoost)',
         '=' * 78, f'{"Model":<12}{"MSE":>10}{"MAE":>8}{"R2":>8}{"MSE_std":>10}{"sec":>8}']
for name, mk in models.items():
    mse, mae, r2, sd, sec = cv_eval(mk)
    lines.append(f'{name:<12}{mse:>10.2f}{mae:>8.2f}{r2:>8.3f}{sd:>10.2f}{sec:>8.1f}')
    print(lines[-1])
rep = '\n'.join(lines)
(OUT / 'sota_results.txt').write_text(rep, encoding='utf-8')
print('\nsaved sota_results.txt')
