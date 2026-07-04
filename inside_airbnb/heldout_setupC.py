"""
heldout_setupC.py -- unbiased evaluation of the best model (Setup C, clean target).
80% development / 20% held-out test split (seed 42).
 - DEV: 5-fold CV OOF -> per-model dev R2 + learn SLSQP blend weights.
 - TEST: refit each model on FULL dev, predict the untouched 20% test -> held-out R2.
Reports dev-CV and held-out-test MSE/R2 for every model and the blend.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
RS, K = 42, 5
df = pd.read_parquet(OUT/'features_setupC.parquet')
y = df['y_clean'].astype(float).values
X = df.drop(columns=['y_clean', 'listing_id']).reset_index(drop=True)
cat = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat}
cat_high = [c for c, n in card.items() if n > 15]; cat_low = [c for c, n in card.items() if n <= 15]
num = X.select_dtypes(include='number').columns.tolist()

dev, test = train_test_split(np.arange(len(y)), test_size=0.20, random_state=RS)
Xdev, ydev, Xtest, ytest = X.iloc[dev].reset_index(drop=True), y[dev], X.iloc[test].reset_index(drop=True), y[test]
print(f'dev n={len(dev)}  held-out test n={len(test)}', flush=True)


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


def pp(scale=False):
    ns=[('imp', SimpleImputer(strategy='median'))]+([('sc', StandardScaler())] if scale else [])
    return ColumnTransformer([('num', Pipeline(ns), num),
        ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
        ('high', KFoldTE(cat_high), cat_high)])

models = {'LinearRegression':(lambda: LinearRegression(), True),
 'RandomForest':(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RS), False),
 'XGBoost':(lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, tree_method='hist'), False),
 'LightGBM':(lambda: LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1), False),
 'CatBoost':(lambda: CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            random_seed=RS, verbose=0), False)}

oof_dev, pred_test, rows = {}, {}, []
for name, (mk, sc) in models.items():
    # dev OOF (5-fold)
    kf = KFold(K, shuffle=True, random_state=RS); oof = np.zeros(len(ydev))
    for tr, va in kf.split(Xdev):
        p = pp(sc); Xtr = p.fit_transform(Xdev.iloc[tr], ydev[tr]); Xva = p.transform(Xdev.iloc[va])
        m = mk(); m.fit(Xtr, ydev[tr]); oof[va] = np.clip(m.predict(Xva), 0, 92)
    oof_dev[name] = oof
    # held-out test: fit preprocessing+model on FULL dev, predict test
    p = pp(sc); Xd = p.fit_transform(Xdev, ydev); Xt = p.transform(Xtest)
    m = mk(); m.fit(Xd, ydev); pt = np.clip(m.predict(Xt), 0, 92); pred_test[name] = pt
    rows.append((name, mean_squared_error(ydev, oof), r2_score(ydev, oof),
                 mean_squared_error(ytest, pt), r2_score(ytest, pt)))
    print(f'{name:<18} devR2={rows[-1][2]:.3f}  TESTR2={rows[-1][4]:.3f}', flush=True)

# blend weights learned on DEV OOF, applied to TEST predictions
names = list(oof_dev); Mdev = np.column_stack([oof_dev[n] for n in names]); Mtest = np.column_stack([pred_test[n] for n in names])
bm = lambda w: mean_squared_error(ydev, np.clip(Mdev@w, 0, 92))
r = minimize(bm, np.full(len(names), 1/len(names)), method='SLSQP', bounds=[(0,1)]*len(names),
             constraints=[{'type':'eq','fun':lambda w: w.sum()-1}], options={'ftol':1e-9, 'maxiter':500})
w = r.x.copy(); w[w<1e-4]=0; w/=w.sum()
bl_dev = np.clip(Mdev@w, 0, 92); bl_test = np.clip(Mtest@w, 0, 92)
rows.append(('SLSQP blend', mean_squared_error(ydev, bl_dev), r2_score(ydev, bl_dev),
             mean_squared_error(ytest, bl_test), r2_score(ytest, bl_test)))

rep = ['HELD-OUT EVALUATION -- Setup C (7-mo history -> Feb-Mar-Apr 2026, clean target)',
       f'dev n={len(dev)}, held-out test n={len(test)}, seed={RS}', '='*68,
       f'{"Model":<18}{"DevMSE":>9}{"DevR2":>8}{"TestMSE":>10}{"TestR2":>9}']
for n, dm, dr, tm, tr in rows:
    rep.append(f'{n:<18}{dm:>9.2f}{dr:>8.3f}{tm:>10.2f}{tr:>9.3f}')
rep.append('blend weights: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names, w)))
txt='\n'.join(rep); print('\n'+txt); (OUT/'heldout_setupC_results.txt').write_text(txt, encoding='utf-8')
