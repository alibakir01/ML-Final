"""XGBoost v3 — no log1p, 60-config tune with fold-level early stopping.

Saves:
  outputs/oof_XGBoostV3.npy
  outputs/submission_xgboost_v3.csv (byte-safe)
  outputs/xgb_v3_best_params.json
  outputs/xgb_v3_search_log.csv
"""
import warnings, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 60
OUT_DIR = Path('outputs')

# ---- Load data ----------------------------------------------------------
# Section IV.C: Use Development portion of split
train_df = pd.read_parquet(OUT_DIR / 'train_dev.parquet')
test_df = pd.read_parquet(OUT_DIR / 'test_features.parquet')
TARGET = 'NumReserveDays2016Q3'; ID = 'PropertyID'
y = train_df[TARGET].astype(float).values
X = train_df.drop(columns=[TARGET, ID]).reset_index(drop=True)
X_test = test_df.drop(columns=[ID]).reindex(columns=X.columns).reset_index(drop=True)
test_ids = test_df[ID].values

cat_all = X.select_dtypes(exclude='number').columns.tolist()
card = {c: X[c].nunique(dropna=False) for c in cat_all}
cat_high = [c for c, n in card.items() if n > 15]
cat_low = [c for c, n in card.items() if n <= 15]
num_cols = X.select_dtypes(include='number').columns.tolist()
print(f'num={len(num_cols)}  cat_low={cat_low}  cat_high={cat_high}')


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42):
        self.cols = cols; self.n_splits = n_splits
        self.smoothing = smoothing; self.random_state = random_state

    def _smap(self, x, y):
        st = pd.DataFrame({'c': x, 'y': y}).groupby('c')['y'].agg(['mean', 'count'])
        return ((st['count'] * st['mean'] + self.smoothing * self.global_mean_)
                / (st['count'] + self.smoothing)).to_dict()

    def fit(self, X, y):
        y = np.asarray(y, dtype=float); self.global_mean_ = float(y.mean())
        self.maps_ = {c: self._smap(X[c].astype(str).fillna('__nan__'), y) for c in self.cols}
        return self

    def transform(self, X):
        Xo = X.copy()
        for c in self.cols:
            Xo[c] = (X[c].astype(str).fillna('__nan__').map(self.maps_[c])
                       .fillna(self.global_mean_).astype('float32'))
        return Xo

    def fit_transform(self, X, y=None, **kw):
        y = np.asarray(y, dtype=float); self.global_mean_ = float(y.mean())
        Xo = X.copy()
        for c in self.cols:
            Xo[c] = np.full(len(X), self.global_mean_, dtype='float32')
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(X):
            for c in self.cols:
                m = self._smap(X[c].astype(str).fillna('__nan__').iloc[tr], y[tr])
                Xo.iloc[va, Xo.columns.get_loc(c)] = (
                    X[c].astype(str).fillna('__nan__').iloc[va].map(m)
                       .fillna(self.global_mean_).astype('float32').values)
        self.maps_ = {c: self._smap(X[c].astype(str).fillna('__nan__'), y) for c in self.cols}
        return Xo


def make_pp():
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_cols),
        ('low', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_low),
        ('high', Pipeline([
            ('te', KFoldTargetEncoder(cols=cat_high, n_splits=5, smoothing=20, random_state=RANDOM_STATE))
        ]), cat_high),
    ])


# ---- CV with fold-level early stopping (NO log1p) ----------------------
outer_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
splits = list(outer_kf.split(X))


def evaluate(params, return_oof=False, return_models=False):
    fold_mses, best_iters, models = [], [], []
    oof = np.zeros(len(y)) if return_oof else None
    for fold, (tr, va) in enumerate(splits):
        pp = make_pp()
        X_tr_full = pp.fit_transform(X.iloc[tr], y[tr])
        X_va = pp.transform(X.iloc[va])
        X_tr, X_in, y_tr, y_in = train_test_split(
            X_tr_full, y[tr], test_size=0.10, random_state=RANDOM_STATE + fold)
        m = XGBRegressor(
            **params,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse',
        )
        m.fit(X_tr, y_tr, eval_set=[(X_in, y_in)], verbose=False)
        pred = np.clip(m.predict(X_va, iteration_range=(0, m.best_iteration + 1)), 0, 92)
        mse = mean_squared_error(y[va], pred)
        fold_mses.append(mse); best_iters.append(m.best_iteration)
        if return_oof: oof[va] = pred
        if return_models: models.append((m, pp, m.best_iteration))
    return {
        'mse_mean': float(np.mean(fold_mses)),
        'mse_std':  float(np.std(fold_mses)),
        'best_iters': best_iters,
        'oof': oof,
        'models': models,
    }


# ---- Random search ------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)

def sample_config():
    return dict(
        n_estimators=3000,
        learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.10)))),
        max_depth=int(rng.integers(4, 9)),
        min_child_weight=int(rng.integers(1, 12)),
        subsample=float(rng.uniform(0.65, 0.95)),
        colsample_bytree=float(rng.uniform(0.65, 0.95)),
        gamma=float(rng.uniform(0.0, 0.4)),
        reg_alpha=float(np.exp(rng.uniform(np.log(1e-3), np.log(0.5)))),
        reg_lambda=float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))),
    )


# Sanity baseline
default_cfg = dict(n_estimators=3000, learning_rate=0.05, max_depth=6, min_child_weight=4,
                   subsample=0.8, colsample_bytree=0.8, gamma=0.0,
                   reg_alpha=0.0, reg_lambda=1.0)
print('\n>>> Sanity baseline (no log1p):')
t0 = time.time(); res = evaluate(default_cfg)
print(f'  Default | CV MSE = {res["mse_mean"]:.3f} ± {res["mse_std"]:.3f} | iters={res["best_iters"]} | {time.time()-t0:.1f}s')

best_score = res['mse_mean']
best_cfg = default_cfg.copy()
log_rows = []

print(f'\n>>> Random search ({N_ITER} configs):')
t_search = time.time()
for i in range(N_ITER):
    cfg = sample_config()
    t0 = time.time()
    res = evaluate(cfg)
    elapsed = time.time() - t0
    log_rows.append({**cfg, 'mse_mean': res['mse_mean'], 'mse_std': res['mse_std'],
                     'avg_best_iter': int(np.mean(res['best_iters'])), 'seconds': elapsed})
    flag = ''
    if res['mse_mean'] < best_score:
        best_score = res['mse_mean']
        best_cfg = cfg.copy()
        flag = '  <- NEW BEST'
    print(f"[{i+1:2d}/{N_ITER}] MSE={res['mse_mean']:7.3f} +- {res['mse_std']:5.3f} "
          f"| iters~{int(np.mean(res['best_iters'])):4d} | lr={cfg['learning_rate']:.4f} "
          f"depth={cfg['max_depth']} | {elapsed:5.1f}s{flag}")

print(f'\nSearch finished in {(time.time()-t_search)/60:.1f} min')
print(f'Best CV MSE: {best_score:.3f}')
print('Best config:', best_cfg)

log_df = pd.DataFrame(log_rows).sort_values('mse_mean').reset_index(drop=True)
log_df.to_csv(OUT_DIR / 'xgb_v3_search_log.csv', index=False)
with open(OUT_DIR / 'xgb_v3_best_params.json', 'w') as f:
    json.dump(best_cfg, f, indent=2)


# ---- Final OOF + test predictions --------------------------------------
print('\n>>> Final 5-fold OOF + test predictions:')
final = evaluate(best_cfg, return_oof=True, return_models=True)
print(f'XGBoostV3 | OOF MSE = {final["mse_mean"]:.3f} +- {final["mse_std"]:.3f}')
np.save(OUT_DIR / 'oof_XGBoostV3.npy', final['oof'])

# Test predictions: average across 5 fold models
test_preds = np.zeros(len(X_test))
for m, pp, best_iter in final['models']:
    Xt = pp.transform(X_test)
    p = np.clip(m.predict(Xt, iteration_range=(0, best_iter + 1)), 0, 92)
    test_preds += p
test_preds /= len(final['models'])
test_preds_int = np.clip(np.round(test_preds), 0, 92).astype(int)

lines = [b'PropertyID_test,Pred\n']
for i, p in zip(test_ids, test_preds_int):
    lines.append(f'{int(i)},{int(p)}\n'.encode('ascii'))
(OUT_DIR / 'submission_xgboost_v3.csv').write_bytes(b''.join(lines))

chk = pd.read_csv(OUT_DIR / 'submission_xgboost_v3.csv')
print(f'\nWrote submission_xgboost_v3.csv | rows={len(chk)} NaN={chk.isna().sum().sum()} '
      f'header={list(chk.columns)} dtype={chk["Pred"].dtype}')
print(chk.head())
print('\nALL DONE.')
