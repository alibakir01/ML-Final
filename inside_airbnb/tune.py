"""
tune.py --data <parquet> --target <col> --tag <name> [--n_iter 30]
Randomized search for XGBoost and LightGBM (log-uniform sampling), 5-fold CV.
Preprocessing (target encoding etc.) is computed ONCE per fold and cached, so the
search only re-fits the model -> fast. Reports best configs and a tuned blend
(tuned XGB + tuned LGBM + existing RF/CatBoost/Linear OOF from oof_<tag>.npz).
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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--target', required=True)
ap.add_argument('--tag', required=True); ap.add_argument('--n_iter', type=int, default=30)
a = ap.parse_args()
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
RS, K = 42, 5
df = pd.read_parquet(a.data)
y = df[a.target].astype(float).values
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

# cache preprocessing per fold
kf = KFold(K, shuffle=True, random_state=RS)
folds = []
for tr, va in kf.split(X):
    pp = make_pp(); Xtr = pp.fit_transform(X.iloc[tr], y[tr]); Xva = pp.transform(X.iloc[va])
    folds.append((np.asarray(Xtr, dtype=np.float32), y[tr], np.asarray(Xva, dtype=np.float32), va))
print('cached', K, 'folds; features:', folds[0][0].shape[1], flush=True)


def cv_oof(make_model):
    oof = np.zeros(len(y))
    for Xtr, ytr, Xva, va in folds:
        m = make_model(); m.fit(Xtr, ytr); oof[va] = np.clip(m.predict(Xva), 0, 92)
    return oof

rng = np.random.default_rng(RS)
def samp_xgb():
    return dict(n_estimators=int(rng.integers(400, 1200)),
                learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
                max_depth=int(rng.integers(4, 10)), min_child_weight=int(rng.integers(1, 12)),
                subsample=float(rng.uniform(0.6, 0.95)), colsample_bytree=float(rng.uniform(0.6, 0.95)),
                reg_lambda=float(np.exp(rng.uniform(np.log(0.5), np.log(8)))),
                reg_alpha=float(np.exp(rng.uniform(np.log(1e-3), np.log(1)))))
def samp_lgb():
    return dict(n_estimators=int(rng.integers(500, 1500)),
                learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
                num_leaves=int(rng.integers(20, 120)), min_child_samples=int(rng.integers(5, 60)),
                subsample=float(rng.uniform(0.6, 0.95)), colsample_bytree=float(rng.uniform(0.6, 0.95)),
                reg_lambda=float(np.exp(rng.uniform(np.log(0.5), np.log(8)))))

best = {}
for fam, samp, ctor in [('XGBoost', samp_xgb,
        lambda c: XGBRegressor(**c, random_state=RS, n_jobs=-1, tree_method='hist')),
        ('LightGBM', samp_lgb,
        lambda c: LGBMRegressor(**c, random_state=RS, n_jobs=-1, verbose=-1))]:
    bmse, bcfg, boof = np.inf, None, None
    for i in range(a.n_iter):
        cfg = samp(); oof = cv_oof(lambda: ctor(cfg)); mse = mean_squared_error(y, oof)
        if mse < bmse: bmse, bcfg, boof = mse, cfg, oof
        print(f'[{fam} {i+1}/{a.n_iter}] MSE={mse:.2f} best={bmse:.2f}', flush=True)
    best[fam] = (bmse, bcfg, boof)
    print(f'>> {fam} best MSE={bmse:.2f}', flush=True)

# tuned blend: tuned XGB + tuned LGBM + existing RF/CatBoost/Linear
prev = np.load(OUT / f'oof_{a.tag}.npz', allow_pickle=True)
comp = {'XGBoost_tuned': best['XGBoost'][2], 'LightGBM_tuned': best['LightGBM'][2],
        'RandomForest': prev['RandomForest'], 'CatBoost': prev['CatBoost'], 'LinearRegression': prev['LinearRegression']}
names = list(comp); M = np.column_stack([comp[n] for n in names])
bm = lambda w: mean_squared_error(y, np.clip(M @ w, 0, 92))
r = minimize(bm, np.full(len(names), 1/len(names)), method='SLSQP', bounds=[(0,1)]*len(names),
             constraints=[{'type':'eq','fun':lambda w: w.sum()-1}], options={'ftol':1e-9, 'maxiter':500})
w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum(); bl = np.clip(M @ w, 0, 92)

rep = [f'TUNING [{a.tag}] target={a.target} ({a.n_iter} configs/family, 5-fold CV)', '='*60,
       f'tuned XGBoost  MSE={best["XGBoost"][0]:.2f}  R2={r2_score(y, best["XGBoost"][2]):.3f}',
       f'tuned LightGBM MSE={best["LightGBM"][0]:.2f}  R2={r2_score(y, best["LightGBM"][2]):.3f}',
       f'tuned blend    MSE={mean_squared_error(y, bl):.2f}  R2={r2_score(y, bl):.3f}',
       'blend weights: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names, w)),
       '', 'best XGBoost cfg: ' + str(best['XGBoost'][1]),
       'best LightGBM cfg: ' + str(best['LightGBM'][1])]
txt = '\n'.join(rep); print('\n' + txt)
(OUT / f'tune_{a.tag}_results.txt').write_text(txt, encoding='utf-8')
np.savez(OUT / f'oof_{a.tag}_tuned.npz', y=y, blend=bl, **comp, weights=w, names=np.array(names))
