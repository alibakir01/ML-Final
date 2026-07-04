"""
heldout_hurdle_setupC.py -- apples-to-apples held-out test of PLAIN BLEND vs
HURDLE+BLEND on the SAME 80/20 split (seed 42), Setup C clean target.
Settles whether the hurdle's +0.002 CV gain survives on unbiased test data.
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
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
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

# cache dev-fold transforms + dev->test transform once (unscaled; linear rescaled separately)
kf = KFold(K, shuffle=True, random_state=RS)
folds_idx = list(kf.split(Xdev))
def dev_oof_and_test(mk, scale=False, proba=False, subset_pos=False):
    oof = np.zeros(len(ydev))
    for tr, va in folds_idx:
        p = pp(scale); Xtr = p.fit_transform(Xdev.iloc[tr], ydev[tr]); Xva = p.transform(Xdev.iloc[va])
        ytr = ydev[tr]
        if subset_pos:
            m = mk(); pos = ytr > 0; m.fit(np.asarray(Xtr)[pos], ytr[pos]); oof[va] = np.clip(m.predict(Xva), 0, 92)
        elif proba:
            m = mk(); m.fit(Xtr, (ytr > 0).astype(int)); oof[va] = m.predict_proba(Xva)[:, 1]
        else:
            m = mk(); m.fit(Xtr, ytr); oof[va] = np.clip(m.predict(Xva), 0, 92)
    p = pp(scale); Xd = p.fit_transform(Xdev, ydev); Xt = p.transform(Xtest)
    if subset_pos:
        m = mk(); pos = ydev > 0; m.fit(np.asarray(Xd)[pos], ydev[pos]); pt = np.clip(m.predict(Xt), 0, 92)
    elif proba:
        m = mk(); m.fit(Xd, (ydev > 0).astype(int)); pt = m.predict_proba(Xt)[:, 1]
    else:
        m = mk(); m.fit(Xd, ydev); pt = np.clip(m.predict(Xt), 0, 92)
    return oof, pt

# ---- single-stage models ----
S = {'LinearRegression':(lambda: LinearRegression(), True),
 'RandomForest':(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RS), False),
 'XGBoost':(lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, tree_method='hist'), False),
 'LightGBM':(lambda: LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1), False),
 'CatBoost':(lambda: CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=RS, verbose=0), False)}
oof_dev, pred_test = {}, {}
for n,(mk,sc) in S.items():
    oof_dev[n], pred_test[n] = dev_oof_and_test(mk, sc); print('single', n, 'done', flush=True)
names=list(oof_dev); Md=np.column_stack([oof_dev[n] for n in names]); Mt=np.column_stack([pred_test[n] for n in names])
r=minimize(lambda w: mean_squared_error(ydev,np.clip(Md@w,0,92)), np.full(len(names),1/len(names)), method='SLSQP',
           bounds=[(0,1)]*len(names), constraints=[{'type':'eq','fun':lambda w:w.sum()-1}], options={'ftol':1e-9})
ws=r.x.copy(); ws[ws<1e-4]=0; ws/=ws.sum()
sb_dev=np.clip(Md@ws,0,92); sb_test=np.clip(Mt@ws,0,92)

# ---- hurdle EV (XGB, LGB) ----
clf_x=lambda: XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, tree_method='hist', eval_metric='logloss')
reg_x=lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, tree_method='hist')
clf_l=lambda: LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=64, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1)
reg_l=lambda: LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1)
px_d,px_t=dev_oof_and_test(clf_x, proba=True); ex_d,ex_t=dev_oof_and_test(reg_x, subset_pos=True)
pl_d,pl_t=dev_oof_and_test(clf_l, proba=True); el_d,el_t=dev_oof_and_test(reg_l, subset_pos=True)
hx_dev=np.clip(px_d*ex_d,0,92); hx_test=np.clip(px_t*ex_t,0,92)
hl_dev=np.clip(pl_d*el_d,0,92); hl_test=np.clip(pl_t*el_t,0,92)
print('hurdle done', flush=True)

# ---- hurdle+single blend (weights on dev) ----
comp_d={'hx':hx_dev,'hl':hl_dev,'sb':sb_dev}; comp_t={'hx':hx_test,'hl':hl_test,'sb':sb_test}
cn=list(comp_d); Hd=np.column_stack([comp_d[n] for n in cn]); Ht=np.column_stack([comp_t[n] for n in cn])
r2=minimize(lambda w: mean_squared_error(ydev,np.clip(Hd@w,0,92)), np.full(3,1/3), method='SLSQP',
            bounds=[(0,1)]*3, constraints=[{'type':'eq','fun':lambda w:w.sum()-1}], options={'ftol':1e-9})
wh=r2.x.copy(); wh[wh<1e-4]=0; wh/=wh.sum()
hb_dev=np.clip(Hd@wh,0,92); hb_test=np.clip(Ht@wh,0,92)

rep=['HELD-OUT: plain blend vs hurdle+blend (Setup C, same 80/20 split, seed 42)',
     f'dev n={len(dev)}, test n={len(test)}', '='*64,
     f'{"Model":<22}{"DevR2":>9}{"TestR2":>9}{"TestMSE":>10}',
     f'{"Plain SLSQP blend":<22}{r2_score(ydev,sb_dev):>9.3f}{r2_score(ytest,sb_test):>9.3f}{mean_squared_error(ytest,sb_test):>10.2f}',
     f'{"Hurdle+blend":<22}{r2_score(ydev,hb_dev):>9.3f}{r2_score(ytest,hb_test):>9.3f}{mean_squared_error(ytest,hb_test):>10.2f}',
     f'hurdle-blend weights: hx={wh[0]:.2f}, hl={wh[1]:.2f}, single_blend={wh[2]:.2f}']
txt='\n'.join(rep); print('\n'+txt); (OUT/'heldout_hurdle_setupC_results.txt').write_text(txt, encoding='utf-8')
