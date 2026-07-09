"""tune_mlp_setupC.py -- EXPERIMENT ONLY. Small hyperparameter search for the PyTorch MLP
on Setup C, to see whether tuning closes the gap to the tree-based learners (R2~0.66).
Searches on a single held-out fold (fast) for ranking, then re-validates the top configs
with the full 5-fold CV used everywhere else in the paper, so the final number is trustworthy.
Run from repo root: python inside_airbnb/tune_mlp_setupC.py
"""
import time, itertools
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
    def fit_transform(self, X, y=None, **kw):
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
    def __init__(self, in_dim, hidden, dropout, use_bn=True):
        super().__init__()
        layers = []; prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval(Xtr, ytr, Xva, yva, cfg, max_epochs=60, patience=10):
    target_tf = cfg.get('target', 'log1p')
    y_tr_raw = ytr if target_tf == 'raw' else np.log1p(ytr)
    X_tr_t = torch.tensor(Xtr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_raw, dtype=torch.float32)
    X_va_t = torch.tensor(Xva, dtype=torch.float32).to(device)
    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=cfg['batch_size'], shuffle=True)
    model = MLPRegressor(Xtr.shape[1], cfg['hidden'], cfg['dropout'], cfg.get('bn', True)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    loss_fn = nn.SmoothL1Loss() if cfg.get('loss') == 'huber' else nn.MSELoss()

    best_mse, best_state, bad = float('inf'), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            raw = model(X_va_t).cpu().numpy()
        pred = raw if target_tf == 'raw' else np.expm1(raw)
        pred = np.clip(pred, 0, 92)
        va_mse = mean_squared_error(yva, pred)
        if va_mse < best_mse - 1e-4:
            best_mse, best_state, bad = va_mse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        raw = model(X_va_t).cpu().numpy()
    pred = raw if target_tf == 'raw' else np.expm1(raw)
    pred = np.clip(pred, 0, 92)
    return mean_squared_error(yva, pred), r2_score(yva, pred), pred


kf = KFold(K, shuffle=True, random_state=RS)
tr0, va0 = next(iter(kf.split(X)))
pp0 = make_pp()
Xtr0 = pp0.fit_transform(X.iloc[tr0], y[tr0]).astype('float32')
Xva0 = pp0.transform(X.iloc[va0]).astype('float32')
yva0 = y[va0]
print('search fold: features =', Xtr0.shape[1])

configs = []
for hidden in [(256, 128, 64), (512, 256, 128), (128, 64), (256, 128, 64, 32), (64, 32)]:
    for dropout in [0.0, 0.1, 0.3]:
        configs.append(dict(hidden=hidden, dropout=dropout, lr=1e-3, wd=1e-5, batch_size=512, target='log1p', loss='mse'))
for lr in [3e-4, 1e-3, 3e-3]:
    for wd in [0.0, 1e-5, 1e-4, 1e-3]:
        configs.append(dict(hidden=(256, 128, 64), dropout=0.2, lr=lr, wd=wd, batch_size=512, target='log1p', loss='mse'))
for target in ['raw', 'log1p']:
    for loss in ['mse', 'huber']:
        for bs in [256, 512, 1024]:
            configs.append(dict(hidden=(256, 128, 64), dropout=0.2, lr=1e-3, wd=1e-5, batch_size=bs, target=target, loss=loss))

# de-dup identical configs
seen, uniq = set(), []
for c in configs:
    key = tuple(sorted(c.items()))
    if key not in seen:
        seen.add(key); uniq.append(c)
configs = uniq
print(f'{len(configs)} configs to search on fold 0\n')

results = []
t0 = time.time()
for i, cfg in enumerate(configs):
    torch.manual_seed(RS)
    mse, r2, _ = train_eval(Xtr0, y[tr0], Xva0, yva0, cfg)
    results.append((r2, mse, cfg))
    print(f'[{i+1:2d}/{len(configs)}] R2={r2:.3f} MSE={mse:.1f}  {cfg}', flush=True)
print(f'\nsearch time: {(time.time()-t0)/60:.1f} min')

results.sort(key=lambda r: -r[0])
print('\nTop 5 configs on search fold:')
for r2, mse, cfg in results[:5]:
    print(f'  R2={r2:.3f} MSE={mse:.1f}  {cfg}')

best_cfg = results[0][2]
print(f'\nRe-validating best config with full 5-fold CV: {best_cfg}')
oof_best = np.zeros(len(y))
for fold, (tr, va) in enumerate(kf.split(X)):
    pp = make_pp()
    Xtr = pp.fit_transform(X.iloc[tr], y[tr]).astype('float32')
    Xva = pp.transform(X.iloc[va]).astype('float32')
    torch.manual_seed(RS)
    _, _, pred = train_eval(Xtr, y[tr], Xva, y[va], best_cfg)
    oof_best[va] = pred
    print(f'  fold {fold+1}/{K} done', flush=True)

mse_b, mae_b, r2_b = mean_squared_error(y, oof_best), mean_absolute_error(y, oof_best), r2_score(y, oof_best)
print(f'\nTuned MLP (5-fold CV, full): MSE={mse_b:.2f} MAE={mae_b:.2f} R2={r2_b:.3f}')

cached = np.load(OUT / 'oof_setupC.npz', allow_pickle=True)
base_names = list(cached['names'])
M_old = np.column_stack([cached[n] for n in base_names])
M_new = np.column_stack([M_old, oof_best])
from scipy.optimize import minimize

def fit_blend(M):
    bm = lambda w: mean_squared_error(y, np.clip(M @ w, 0, 92))
    r = minimize(bm, np.full(M.shape[1], 1 / M.shape[1]), method='SLSQP', bounds=[(0, 1)] * M.shape[1],
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-9, 'maxiter': 500})
    w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum()
    bl = np.clip(M @ w, 0, 92)
    return w, mean_squared_error(y, bl), r2_score(y, bl)

w_old, mse_old, r2_old = fit_blend(M_old)
w_new, mse_new, r2_new = fit_blend(M_new)
names_new = base_names + ['MLP_tuned']
print(f'\nBlend without MLP: R2={r2_old:.3f}')
print(f'Blend with tuned MLP: R2={r2_new:.3f}  weights=' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names_new, w_new)))
print(f'Delta R2: {r2_new - r2_old:+.4f}')

(OUT / 'tune_mlp_setupC_results.txt').write_text(
    f'Best config: {best_cfg}\nTuned MLP solo: MSE={mse_b:.2f} MAE={mae_b:.2f} R2={r2_b:.3f}\n'
    f'Blend without MLP: R2={r2_old:.3f}\nBlend with tuned MLP: R2={r2_new:.3f} (delta {r2_new-r2_old:+.4f})\n'
    f'weights with tuned MLP: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names_new, w_new)),
    encoding='utf-8')
np.savez(OUT / 'oof_tuned_mlp_setupC.npz', y=y, MLP_tuned=oof_best)
print('\nsaved outputs/tune_mlp_setupC_results.txt')
