"""
model_run_timed.py --data <parquet> --target <col> --tag <name>
Same generic pipeline as model_run.py (median impute + one-hot low-card + K-fold
target-encode high-card, 5-fold CV, clip [0,92]) for Linear/RF/XGBoost/LightGBM/
CatBoost + SLSQP blend, but times each model's cumulative 5-fold-CV wall clock and
the blend fit, and writes a Time(s) column to the results file (matching the format
of outputs/heldout_setupC_with_mlp_results.txt).
Run from repo root: python inside_airbnb/model_run_timed.py --data ... --target ... --tag ...
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--target', required=True)
ap.add_argument('--tag', required=True)
a = ap.parse_args()
OUT = Path('inside_airbnb/outputs')
RS, K = 42, 5
df = pd.read_parquet(a.data)
y = df[a.target].astype(float).values
X = df.drop(columns=[a.target, 'listing_id']).reset_index(drop=True)
cat = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat}
cat_high = [c for c, n in card.items() if n > 15]; cat_low = [c for c, n in card.items() if n <= 15]
num = X.select_dtypes(include='number').columns.tolist()
print(f'n={len(y)}, {len(num)} numeric + {len(cat_low)} low-card + {len(cat_high)} high-card '
      f'({len(num)+len(cat_low)+len(cat_high)} raw feature columns)', flush=True)


class KFoldTE(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        self.cols=cols; self.n_splits=n_splits; self.smoothing=smoothing; self.random_state=random_state
    def _m(self, x, yy):
        st = pd.DataFrame({'c': x, 'y': yy}).groupby('c')['y'].agg(['mean', 'count'])
        return ((st['count']*st['mean']+self.smoothing*self.gm_)/(st['count']+self.smoothing)).to_dict()
    def fit(self, X, y):
        y=np.asarray(y,float); self.gm_=float(y.mean())
        self.maps_={c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return self
    def transform(self, X):
        Xo=X[self.cols].copy()
        for c in self.cols: Xo[c]=X[c].astype(str).fillna('_n').map(self.maps_[c]).fillna(self.gm_).astype('float32')
        return Xo
    def fit_transform(self, X, y=None, **k):
        y=np.asarray(y,float); self.gm_=float(y.mean()); Xo=X[self.cols].copy()
        for c in self.cols: Xo[c]=np.full(len(X), self.gm_, 'float32')
        kk=KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kk.split(X):
            for c in self.cols:
                m=self._m(X[c].astype(str).fillna('_n').iloc[tr], y[tr])
                Xo.iloc[va, Xo.columns.get_loc(c)]=X[c].astype(str).fillna('_n').iloc[va].map(m).fillna(self.gm_).astype('float32').values
        self.maps_={c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return Xo


def pp(scale=False):
    ns=[('imp', SimpleImputer(strategy='median'))]+([('sc', StandardScaler())] if scale else [])
    return ColumnTransformer([('num', Pipeline(ns), num),
        ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
        ('high', KFoldTE(cat_high), cat_high)])


def cv(mk, scale=False):
    kf=KFold(K, shuffle=True, random_state=RS); oof=np.zeros(len(y))
    for tr, va in kf.split(X):
        p=pp(scale); Xtr=p.fit_transform(X.iloc[tr], y[tr]); Xva=p.transform(X.iloc[va])
        m=mk(); m.fit(Xtr, y[tr]); oof[va]=np.clip(m.predict(Xva), 0, 92)
    return oof

models={'LinearRegression':(lambda: LinearRegression(), True),
 'RandomForest':(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RS), False),
 'XGBoost':(lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, tree_method='hist'), False),
 'LightGBM':(lambda: LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1), False),
 'CatBoost':(lambda: CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            random_seed=RS, verbose=0), False)}
oofs, rows, times = {}, [], {}
for n, (mk, sc) in models.items():
    t0 = time.time()
    o=cv(mk, sc); oofs[n]=o
    times[n] = time.time() - t0
    rows.append((n, mean_squared_error(y,o), mean_absolute_error(y,o), r2_score(y,o)))
    print(f'{n:<18} MSE={rows[-1][1]:7.2f} MAE={rows[-1][2]:6.2f} R2={rows[-1][3]:.3f}  time={times[n]:.1f}s', flush=True)

names=list(oofs); M=np.column_stack([oofs[n] for n in names])
t0 = time.time()
bm=lambda w: mean_squared_error(y, np.clip(M@w,0,92))
r=minimize(bm, np.full(len(names),1/len(names)), method='SLSQP', bounds=[(0,1)]*len(names),
           constraints=[{'type':'eq','fun':lambda w: w.sum()-1}], options={'ftol':1e-9,'maxiter':500})
w=r.x.copy(); w[w<1e-4]=0; w/=w.sum(); bl=np.clip(M@w,0,92)
times['SLSQP blend'] = time.time() - t0
rows.append(('SLSQP blend', mean_squared_error(y,bl), mean_absolute_error(y,bl), r2_score(y,bl)))
np.savez(OUT/f'oof_{a.tag}.npz', y=y, **{n: oofs[n] for n in names}, blend=bl, weights=w, names=np.array(names))
rep=[f'RESULTS [{a.tag}] target={a.target} (5-fold CV, n={len(y)})', '='*68,
     f'{"Model":<18}{"MSE":>9}{"MAE":>8}{"R2":>8}{"Time(s)":>10}']
rep += [f'{n:<18}{mse:>9.2f}{mae:>8.2f}{r2:>8.3f}{times[n]:>10.1f}' for n,mse,mae,r2 in rows]
rep.append('\nBlend weights: '+', '.join(f'{n}={wi:.3f}' for n,wi in zip(names,w)))
txt='\n'.join(rep); print('\n'+txt)
(OUT/f'model_{a.tag}_results.txt').write_text(txt, encoding='utf-8')
