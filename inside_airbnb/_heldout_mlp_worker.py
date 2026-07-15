"""_heldout_mlp_worker.py -- runs exactly ONE model family for the Setup C held-out
evaluation with MLP, in its own process. This avoids the macOS crash (SIGSEGV inside
libomp's __kmp_suspend_64) that occurs when XGBoost and LightGBM's separately-bundled
OpenMP runtimes are both loaded in the same process.
Usage: python _heldout_mlp_worker.py <linear|rf|xgboost|lightgbm|catboost|mlp>
Writes outputs/_heldout_mlp_part_<name>.npz with dev_oof, test_pred, seconds.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

OUT = Path('inside_airbnb/outputs')
RS, K = 42, 5
name = sys.argv[1]

df = pd.read_parquet(OUT / 'features_setupC.parquet')
y = df['y_clean'].astype(float).values
X = df.drop(columns=['y_clean', 'listing_id']).reset_index(drop=True)
cat = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]
num = X.select_dtypes(include='number').columns.tolist()

dev, test = train_test_split(np.arange(len(y)), test_size=0.20, random_state=RS)
Xdev, ydev, Xtest, ytest = X.iloc[dev].reset_index(drop=True), y[dev], X.iloc[test].reset_index(drop=True), y[test]


class KFoldTE(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        self.cols = cols; self.n_splits = n_splits; self.smoothing = smoothing; self.random_state = random_state
    def _m(self, x, yy):
        st = pd.DataFrame({'c': x, 'y': yy}).groupby('c')['y'].agg(['mean', 'count'])
        return ((st['count'] * st['mean'] + self.smoothing * self.gm_) / (st['count'] + self.smoothing)).to_dict()
    def fit(self, X, y):
        y = np.asarray(y, float); self.gm_ = float(y.mean())
        self.maps_ = {c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return self
    def transform(self, X):
        Xo = X[self.cols].copy()
        for c in self.cols:
            Xo[c] = X[c].astype(str).fillna('_n').map(self.maps_[c]).fillna(self.gm_).astype('float32')
        return Xo
    def fit_transform(self, X, y=None, **k):
        y = np.asarray(y, float); self.gm_ = float(y.mean()); Xo = X[self.cols].copy()
        for c in self.cols:
            Xo[c] = np.full(len(X), self.gm_, 'float32')
        kk = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kk.split(X):
            for c in self.cols:
                m = self._m(X[c].astype(str).fillna('_n').iloc[tr], y[tr])
                Xo.iloc[va, Xo.columns.get_loc(c)] = X[c].astype(str).fillna('_n').iloc[va].map(m).fillna(self.gm_).astype('float32').values
        self.maps_ = {c: self._m(X[c].astype(str).fillna('_n'), y) for c in self.cols}; return Xo


def pp(scale=False, mlp=False):
    ns = [('imp', SimpleImputer(strategy='median', add_indicator=mlp))] + ([('sc', StandardScaler())] if (scale or mlp) else [])
    high = Pipeline([('te', KFoldTE(cat_high)), ('sc', StandardScaler())]) if mlp else KFoldTE(cat_high)
    return ColumnTransformer([('num', Pipeline(ns), num),
        ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
        ('high', high, cat_high)])


def run_sklearn_model(mk, scale):
    kf = KFold(K, shuffle=True, random_state=RS); oof = np.zeros(len(ydev))
    for tr, va in kf.split(Xdev):
        p = pp(scale); Xtr = p.fit_transform(Xdev.iloc[tr], ydev[tr]); Xva = p.transform(Xdev.iloc[va])
        m = mk(); m.fit(Xtr, ydev[tr]); oof[va] = np.clip(m.predict(Xva), 0, 92)
    p = pp(scale); Xd = p.fit_transform(Xdev, ydev); Xt = p.transform(Xtest)
    m = mk(); m.fit(Xd, ydev); pt = np.clip(m.predict(Xt), 0, 92)
    return oof, pt


t0 = time.time()

if name == 'linear':
    from sklearn.linear_model import LinearRegression
    oof, pt = run_sklearn_model(lambda: LinearRegression(), True)
elif name == 'rf':
    from sklearn.ensemble import RandomForestRegressor
    oof, pt = run_sklearn_model(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RS), False)
elif name == 'xgboost':
    from xgboost import XGBRegressor
    oof, pt = run_sklearn_model(lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, tree_method='hist'), False)
elif name == 'lightgbm':
    from lightgbm import LGBMRegressor
    oof, pt = run_sklearn_model(lambda: LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=-1, verbose=-1), False)
elif name == 'catboost':
    from catboost import CatBoostRegressor
    oof, pt = run_sklearn_model(lambda: CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        random_seed=RS, verbose=0), False)
elif name == 'mlp':
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(RS); np.random.seed(RS)
    device = torch.device('mps') if torch.backends.mps.is_available() else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'PyTorch {torch.__version__} on {device}', flush=True)

    class MLPRegressor(nn.Module):
        def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.2):
            super().__init__()
            layers = []; prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_predict(Xtr, ytr, Xva, yva_for_stopping, max_epochs=60, batch_size=256, lr=1e-3, weight_decay=1e-5, patience=10):
        X_tr_t = torch.tensor(Xtr, dtype=torch.float32)
        y_tr_t = torch.tensor(ytr, dtype=torch.float32)
        X_va_t = torch.tensor(Xva, dtype=torch.float32).to(device)
        dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        model = MLPRegressor(Xtr.shape[1]).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        best_mse, best_state, bad = float('inf'), None, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optim.step()
            model.eval()
            with torch.no_grad():
                pred = np.clip(model(X_va_t).cpu().numpy(), 0, 92)
            if yva_for_stopping is not None:
                va_mse = mean_squared_error(yva_for_stopping, pred)
                if va_mse < best_mse - 1e-4:
                    best_mse, best_state, bad = va_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final = np.clip(model(X_va_t).cpu().numpy(), 0, 92)
        return final

    kf = KFold(K, shuffle=True, random_state=RS); oof = np.zeros(len(ydev))
    for fold, (tr, va) in enumerate(kf.split(Xdev)):
        p = pp(mlp=True)
        Xtr = p.fit_transform(Xdev.iloc[tr], ydev[tr]).astype('float32')
        Xva = p.transform(Xdev.iloc[va]).astype('float32')
        torch.manual_seed(RS)
        oof[va] = train_predict(Xtr, ydev[tr], Xva, ydev[va])
        print(f'  MLP fold {fold+1}/{K} done', flush=True)

    p = pp(mlp=True)
    Xd = p.fit_transform(Xdev, ydev).astype('float32'); Xt = p.transform(Xtest).astype('float32')
    tr_idx, va_idx = train_test_split(np.arange(len(ydev)), test_size=0.1, random_state=RS)
    # an internal 90/10 split of dev provides validation labels for early stopping;
    # the resulting model (selected on that internal 10%) predicts the held-out test set
    X_tr_t = torch.tensor(Xd[tr_idx], dtype=torch.float32)
    y_tr_t = torch.tensor(ydev[tr_idx], dtype=torch.float32)
    X_va_t = torch.tensor(Xd[va_idx], dtype=torch.float32).to(device)
    X_test_t = torch.tensor(Xt, dtype=torch.float32).to(device)
    torch.manual_seed(RS)
    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)
    model = MLPRegressor(Xd.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_mse, best_state, bad = float('inf'), None, 0
    for epoch in range(1, 61):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            va_pred = np.clip(model(X_va_t).cpu().numpy(), 0, 92)
        va_mse = mean_squared_error(ydev[va_idx], va_pred)
        if va_mse < best_mse - 1e-4:
            best_mse, best_state, bad = va_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 10:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = np.clip(model(X_test_t).cpu().numpy(), 0, 92)
else:
    raise SystemExit(f'unknown model {name}')

elapsed = time.time() - t0
np.savez(OUT / f'_heldout_mlp_part_{name}.npz', dev_oof=oof, test_pred=pt, seconds=elapsed,
         ydev=ydev, ytest=ytest)
from sklearn.metrics import r2_score
print(f'{name}: devR2={r2_score(ydev, oof):.3f} testR2={r2_score(ytest, pt):.3f} ({elapsed:.1f}s)')
