"""
hurdle.py --data <parquet> --target <col> --tag <name>
Two-stage hurdle model for the zero-inflated target:
  Stage 1: classifier  p(x) = P(y>0 | x)
  Stage 2: regressor    r(x) = E[y | y>0, x]   (trained on positive rows only)
  prediction (expected value): y_hat = clip(p(x) * r(x), 0, 92)
Built for XGBoost and LightGBM; compared to the single-stage OOF blend, and a
final blend of {hurdle_xgb, hurdle_lgbm, single-stage models} is optimized.
Preprocessing (target encoding) cached per fold. Saves results + oof.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--target', required=True); ap.add_argument('--tag', required=True)
a = ap.parse_args()
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
RS, K = 42, 5
df = pd.read_parquet(a.data); y = df[a.target].astype(float).values
X = df.drop(columns=[a.target, 'listing_id']).reset_index(drop=True)
cat = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat}
cat_high = [c for c, n in card.items() if n > 15]; cat_low = [c for c, n in card.items() if n <= 15]
num = X.select_dtypes(include='number').columns.tolist()


class KFoldTE(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        self.cols=cols; self.n_splits=n_splits; self.smoothing=smoothing; self.random_state=random_state
    def _m(self, x, yy):
        st = pd.DataFrame({'c': x, 'y': yy}).groupby('c')['y'].agg(['mean','count'])
        return ((st['count']*st['mean']+self.smoothing*self.gm_)/(st['count']+self.smoothing)).to_dict()
    def fit(self, X, y):
        y=np.asarray(y,float); self.gm_=float(y.mean())
        self.maps_={c:self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return self
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
        self.maps_={c:self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return Xo


def make_pp():
    return ColumnTransformer([('num', SimpleImputer(strategy='median'), num),
        ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
        ('high', KFoldTE(cat_high), cat_high)])

kf = KFold(K, shuffle=True, random_state=RS)
folds = []
for tr, va in kf.split(X):
    pp = make_pp(); Xtr = np.asarray(pp.fit_transform(X.iloc[tr], y[tr]), np.float32)
    Xva = np.asarray(pp.transform(X.iloc[va]), np.float32)
    folds.append((Xtr, y[tr], Xva, va))
print('cached folds', flush=True)

XGB_R = dict(n_estimators=759, learning_rate=0.016, max_depth=8, min_child_weight=4, subsample=0.80,
             colsample_bytree=0.66, reg_lambda=5.4, reg_alpha=0.19, random_state=RS, n_jobs=-1, tree_method='hist')
LGB_R = dict(n_estimators=900, learning_rate=0.025, num_leaves=80, subsample=0.8, colsample_bytree=0.7,
             reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1)

def hurdle_oof(clf_ctor, reg_ctor):
    oof_ev = np.zeros(len(y)); oof_gate = np.zeros(len(y))
    for Xtr, ytr, Xva, va in folds:
        pos = ytr > 0
        clf = clf_ctor(); clf.fit(Xtr, pos.astype(int))
        p = clf.predict_proba(Xva)[:, 1]
        reg = reg_ctor(); reg.fit(Xtr[pos], ytr[pos])
        e = np.clip(reg.predict(Xva), 0, 92)
        oof_ev[va] = np.clip(p * e, 0, 92)
        oof_gate[va] = np.where(p >= 0.5, e, 0.0)
    return oof_ev, oof_gate

res = {}
xgb_ev, xgb_gate = hurdle_oof(
    lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=7, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1,
                          tree_method='hist', eval_metric='logloss'),
    lambda: XGBRegressor(**XGB_R))
print('xgb hurdle done', flush=True)
lgb_ev, lgb_gate = hurdle_oof(
    lambda: LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=64, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1),
    lambda: LGBMRegressor(**LGB_R))
print('lgb hurdle done', flush=True)

for nm, oof in [('hurdle_xgb_EV', xgb_ev), ('hurdle_xgb_gate', xgb_gate),
                ('hurdle_lgb_EV', lgb_ev), ('hurdle_lgb_gate', lgb_gate)]:
    res[nm] = (mean_squared_error(y, oof), r2_score(y, oof))

# blend: hurdle EVs + prior single-stage tuned OOFs (if available)
comp = {'hurdle_xgb_EV': xgb_ev, 'hurdle_lgb_EV': lgb_ev}
for cand in [OUT / f'oof_{a.tag}_tuned.npz', OUT / f'oof_{a.tag}.npz']:
    if cand.exists():
        comp['single_blend'] = np.load(cand, allow_pickle=True)['blend']; break
names = list(comp); M = np.column_stack([comp[n] for n in names])
bm = lambda w: mean_squared_error(y, np.clip(M @ w, 0, 92))
r = minimize(bm, np.full(len(names), 1/len(names)), method='SLSQP', bounds=[(0,1)]*len(names),
             constraints=[{'type':'eq','fun':lambda w: w.sum()-1}], options={'ftol':1e-9, 'maxiter':500})
w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum(); bl = np.clip(M @ w, 0, 92)
res['HURDLE+single blend'] = (mean_squared_error(y, bl), r2_score(y, bl))

rep = [f'HURDLE [{a.tag}] target={a.target} (5-fold CV, n={len(y)})', '='*58, f'{"Variant":<24}{"MSE":>9}{"R2":>8}']
for nm, (mse, r2) in res.items(): rep.append(f'{nm:<24}{mse:>9.2f}{r2:>8.3f}')
rep.append('blend weights: ' + ', '.join(f'{n}={wi:.2f}' for n, wi in zip(names, w)))
txt = '\n'.join(rep); print('\n'+txt); (OUT/f'hurdle_{a.tag}_results.txt').write_text(txt, encoding='utf-8')
