"""_mlp_cv_worker.py -- 5-fold CV OOF for the tuned PyTorch MLP on an arbitrary Setup's
feature parquet, run in its own process (see heldout_setupC_with_mlp.py for why: XGBoost
and LightGBM's separately-bundled OpenMP runtimes crash when PyTorch is also loaded in
the same process). Same tuned config as Setup C (raw target, 256-128-64, dropout 0.2,
lr=1e-3, wd=1e-5, batch_size=256).
Usage: python _mlp_cv_worker.py <parquet_path> <target_col> <output_tag>
Writes outputs/oof_mlp_only_<tag>.npz with y, mlp_oof.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

OUT = Path('inside_airbnb/outputs')
RS, K = 42, 5
parquet_path, target_col, tag = sys.argv[1], sys.argv[2], sys.argv[3]

torch.manual_seed(RS); np.random.seed(RS)
device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f'PyTorch {torch.__version__} on {device}', flush=True)

df = pd.read_parquet(parquet_path)
y = df[target_col].astype(float).values
X = df.drop(columns=[target_col, 'listing_id']).reset_index(drop=True)
cat = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]
num = X.select_dtypes(include='number').columns.tolist()
print(f'n={len(y)}, {len(num)} numeric + {len(cat_low)} low-card + {len(cat_high)} high-card', flush=True)


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


def make_pp():
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)), ('sc', StandardScaler())])
    low_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    high_pipe = Pipeline([('te', KFoldTE(cat_high, random_state=RS)), ('sc', StandardScaler())])
    return ColumnTransformer([('num', num_pipe, num), ('low', low_pipe, cat_low), ('high', high_pipe, cat_high)])


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


def train_predict(Xtr, ytr, Xva, yva, max_epochs=60, batch_size=256, lr=1e-3, weight_decay=1e-5, patience=10):
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
        va_mse = mean_squared_error(yva, pred)
        if va_mse < best_mse - 1e-4:
            best_mse, best_state, bad = va_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        final = np.clip(model(X_va_t).cpu().numpy(), 0, 92)
    return final


t0 = time.time()
kf = KFold(K, shuffle=True, random_state=RS); oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(kf.split(X)):
    pp = make_pp()
    Xtr = pp.fit_transform(X.iloc[tr], y[tr]).astype('float32')
    Xva = pp.transform(X.iloc[va]).astype('float32')
    torch.manual_seed(RS)
    oof[va] = train_predict(Xtr, y[tr], Xva, y[va])
    print(f'  fold {fold+1}/{K} done', flush=True)
elapsed = time.time() - t0

mse, r2 = mean_squared_error(y, oof), r2_score(y, oof)
print(f'{tag}: MLP 5-fold CV MSE={mse:.2f} R2={r2:.3f} ({elapsed:.1f}s)')
np.savez(OUT / f'oof_mlp_only_{tag}.npz', y=y, mlp_oof=oof, seconds=elapsed)
print(f'saved outputs/oof_mlp_only_{tag}.npz')
