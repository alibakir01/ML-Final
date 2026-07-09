"""experiment_mlp_setupC.py -- EXPERIMENT ONLY, not wired into the paper's reported results.

Trains a PyTorch MLP (same architecture as the old notebooks/06_MLP_PyTorch.ipynb: 3 hidden
layers 256->128->64, BatchNorm+ReLU+Dropout, Adam, log1p target) on the Setup C feature matrix
(features_setupC.parquet, target y_clean), using the *exact same* KFold(5, shuffle=True,
random_state=42) split that produced outputs/oof_setupC.npz (model_run.py), so its OOF
predictions are fold-aligned with the five existing base learners.

Reports:
  1. the MLP's own solo 5-fold CV MSE/MAE/R2
  2. whether adding it as a 6th learner to the SLSQP convex blend improves on the current
     blend R2=0.664 (n=36,282, CV) reported in the paper's Table IX.

Run from repo root: python inside_airbnb/experiment_mlp_setupC.py
"""
import time
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize

OUT = Path('inside_airbnb/outputs')
RS, K = 42, 5
torch.manual_seed(RS); np.random.seed(RS)
device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f'PyTorch {torch.__version__} on {device}')

df = pd.read_parquet(OUT / 'features_setupC.parquet')
y = df['y_clean'].astype(float).values
X = df.drop(columns=['y_clean', 'listing_id']).reset_index(drop=True)
cat = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].astype(str).nunique() for c in cat}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]
num = X.select_dtypes(include='number').columns.tolist()


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


def train_one_fold(X_tr, y_tr, X_va, y_va, *, max_epochs=60, batch_size=512, lr=1e-3,
                    weight_decay=1e-5, patience=8):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(np.log1p(y_tr), dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_arr = np.asarray(y_va, dtype=float)

    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    model = MLPRegressor(in_dim=X_tr.shape[1]).to(device)
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
            pred_va = np.clip(np.expm1(model(X_va_t).cpu().numpy()), 0, 92)
        va_mse = mean_squared_error(y_va_arr, pred_va)
        if va_mse < best_mse - 1e-4:
            best_mse, best_state, bad = va_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        final_va = np.clip(np.expm1(model(X_va_t).cpu().numpy()), 0, 92)
    return best_mse, final_va


print(f'n={len(y)}, {len(num)} numeric + {len(cat_low)} low-card + {len(cat_high)} high-card categorical cols')
kf = KFold(K, shuffle=True, random_state=RS)
oof_mlp = np.zeros(len(y))
t0 = time.time()
for fold, (tr, va) in enumerate(kf.split(X)):
    pp = make_pp()
    Xtr = pp.fit_transform(X.iloc[tr], y[tr]).astype('float32')
    Xva = pp.transform(X.iloc[va]).astype('float32')
    fold_mse, pred_va = train_one_fold(Xtr, y[tr], Xva, y[va])
    oof_mlp[va] = pred_va
    print(f'  fold {fold+1}/{K}: {Xtr.shape[1]} features, val MSE={fold_mse:.2f}', flush=True)
print(f'MLP total time: {(time.time()-t0)/60:.1f} min')

mlp_mse, mlp_mae, mlp_r2 = mean_squared_error(y, oof_mlp), mean_absolute_error(y, oof_mlp), r2_score(y, oof_mlp)
print(f'\nMLP solo (5-fold CV): MSE={mlp_mse:.2f}  MAE={mlp_mae:.2f}  R2={mlp_r2:.3f}')

# ---- does adding it to the blend help? reuse the cached 5-model OOF (fold-aligned KFold(42)) ----
cached = np.load(OUT / 'oof_setupC.npz', allow_pickle=True)
assert np.allclose(cached['y'], y), 'row order mismatch vs cached oof_setupC.npz'
base_names = list(cached['names'])
M_old = np.column_stack([cached[n] for n in base_names])
names_new = base_names + ['MLP']
M_new = np.column_stack([M_old, oof_mlp])

def fit_blend(M):
    bm = lambda w: mean_squared_error(y, np.clip(M @ w, 0, 92))
    r = minimize(bm, np.full(M.shape[1], 1 / M.shape[1]), method='SLSQP', bounds=[(0, 1)] * M.shape[1],
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-9, 'maxiter': 500})
    w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum()
    bl = np.clip(M @ w, 0, 92)
    return w, mean_squared_error(y, bl), r2_score(y, bl)

w_old, mse_old, r2_old = fit_blend(M_old)
w_new, mse_new, r2_new = fit_blend(M_new)

print(f'\nBlend WITHOUT MLP: R2={r2_old:.3f} MSE={mse_old:.2f}  weights=' +
      ', '.join(f'{n}={wi:.3f}' for n, wi in zip(base_names, w_old)))
print(f'Blend WITH MLP:    R2={r2_new:.3f} MSE={mse_new:.2f}  weights=' +
      ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names_new, w_new)))
print(f'\nDelta R2 from adding MLP to the blend: {r2_new - r2_old:+.4f}')

rep = [
    'EXPERIMENT: PyTorch MLP as a 6th base learner, Setup C (5-fold CV, n=%d)' % len(y),
    '=' * 68,
    f'MLP solo:            MSE={mlp_mse:.2f}  MAE={mlp_mae:.2f}  R2={mlp_r2:.3f}',
    f'Blend without MLP:   MSE={mse_old:.2f}  R2={r2_old:.3f}',
    f'Blend with MLP:      MSE={mse_new:.2f}  R2={r2_new:.3f}  (delta R2 = {r2_new - r2_old:+.4f})',
    'weights without MLP: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(base_names, w_old)),
    'weights with MLP:    ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names_new, w_new)),
]
(OUT / 'experiment_mlp_setupC_results.txt').write_text('\n'.join(rep), encoding='utf-8')
np.savez(OUT / 'oof_experiment_mlp_setupC.npz', y=y, MLP=oof_mlp)
print('\nsaved outputs/experiment_mlp_setupC_results.txt')
