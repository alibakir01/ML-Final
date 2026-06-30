"""
model_setupA.py
Replicate the competition pipeline on the Inside Airbnb Setup A data:
predict Q4 2025 booked nights from Q3 2025 features. Same leakage-safe
preprocessing (median impute + one-hot low-card + K-fold target encode high-card),
5-fold CV (random_state=42), clip [0,92]. Models: Linear, RandomForest, XGBoost,
LightGBM, CatBoost. Reports MSE/MAE/R2 to outputs/model_setupA_results.txt
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
RS, K = 42, 5
df = pd.read_parquet(OUT / 'features_setupA.parquet')
y = df['y_booked_q4'].astype(float).values
X = df.drop(columns=['y_booked_q4', 'listing_id']).reset_index(drop=True)

cat_all = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat_all}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]
num_cols = X.select_dtypes(include='number').columns.tolist()
print('num:', len(num_cols), '| low-card cat:', cat_low, '| high-card cat:', cat_high)


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
        return Xo[self.cols]
    def fit_transform(self, X, y=None, **k):
        y = np.asarray(y, float); self.gm_ = float(y.mean()); Xo = X[self.cols].copy()
        for c in self.cols: Xo[c] = np.full(len(X), self.gm_, 'float32')
        kk = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kk.split(X):
            for c in self.cols:
                m = self._m(X[c].astype(str).fillna('_n').iloc[tr], y[tr])
                Xo.iloc[va, Xo.columns.get_loc(c)] = X[c].astype(str).fillna('_n').iloc[va].map(m).fillna(self.gm_).astype('float32').values
        self.maps_ = {c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return Xo


def make_pp(scale=False):
    num_steps = [('imp', SimpleImputer(strategy='median'))] + ([('sc', StandardScaler())] if scale else [])
    return ColumnTransformer([
        ('num', Pipeline(num_steps), num_cols),
        ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
        ('high', KFoldTE(cat_high), cat_high)])


def cv(make_model, scale=False):
    kf = KFold(K, shuffle=True, random_state=RS); oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        pp = make_pp(scale)
        Xtr = pp.fit_transform(X.iloc[tr], y[tr]); Xva = pp.transform(X.iloc[va])
        m = make_model(); m.fit(Xtr, y[tr])
        oof[va] = np.clip(m.predict(Xva), 0, 92)
    return oof

models = {
    'LinearRegression': (lambda: LinearRegression(), True),
    'RandomForest': (lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RS), False),
    'XGBoost': (lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS,
                n_jobs=-1, tree_method='hist'), False),
    'LightGBM': (lambda: LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1), False),
    'CatBoost': (lambda: CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6,
                l2_leaf_reg=3.0, random_seed=RS, verbose=0), False),
}
oofs, rows = {}, []
for name, (mk, sc) in models.items():
    oof = cv(mk, sc); oofs[name] = oof
    rows.append((name, mean_squared_error(y, oof), mean_absolute_error(y, oof), r2_score(y, oof)))
    print(f'{name:<18} MSE={rows[-1][1]:7.2f} MAE={rows[-1][2]:6.2f} R2={rows[-1][3]:.3f}')

# SLSQP convex blend
from scipy.optimize import minimize
names = list(oofs); M = np.column_stack([oofs[n] for n in names])
def bm(w): return mean_squared_error(y, np.clip(M @ w, 0, 92))
r = minimize(bm, np.full(len(names), 1/len(names)), method='SLSQP',
             bounds=[(0, 1)]*len(names), constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
             options={'ftol': 1e-9, 'maxiter': 500})
w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum()
blend = np.clip(M @ w, 0, 92)
rows.append(('SLSQP blend', mean_squared_error(y, blend), mean_absolute_error(y, blend), r2_score(y, blend)))

rep = ['SETUP A RESULTS -- predict Q4 2025 booked nights (5-fold CV, n=%d)' % len(y),
       '=' * 60, f'{"Model":<18}{"MSE":>9}{"MAE":>8}{"R2":>8}']
for n, mse, mae, r2 in rows:
    rep.append(f'{n:<18}{mse:>9.2f}{mae:>8.2f}{r2:>8.3f}')
rep.append('\nBlend weights: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names, w)))
out = '\n'.join(rep)
print('\n' + out)
(OUT / 'model_setupA_results.txt').write_text(out, encoding='utf-8')
print('\nsaved model_setupA_results.txt')
