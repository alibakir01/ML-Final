"""heldout_setupC_with_mlp.py -- orchestrator. Runs each model family for the Setup C
held-out evaluation (5-fold CV on dev + fit-on-full-dev/predict-test) in its OWN
subprocess via _heldout_mlp_worker.py, then combines the results into a 6-learner
SLSQP blend. Each model gets its own process because XGBoost and LightGBM each bundle
a separate copy of libomp on macOS; loading both in one process segfaults
(SIGSEGV inside __kmp_suspend_64) once a third heavy native library (PyTorch) is
also active. Isolating processes sidesteps the conflict entirely.
Run from repo root: python inside_airbnb/heldout_setupC_with_mlp.py
"""
import subprocess, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

OUT = Path('inside_airbnb/outputs')
MODELS = ['linear', 'rf', 'xgboost', 'lightgbm', 'catboost', 'mlp']
DISPLAY = {'linear': 'LinearRegression', 'rf': 'RandomForest', 'xgboost': 'XGBoost',
           'lightgbm': 'LightGBM', 'catboost': 'CatBoost', 'mlp': 'MLP'}

for name in MODELS:
    part_file = OUT / f'_heldout_mlp_part_{name}.npz'
    if part_file.exists():
        print(f'{name}: already computed, skipping', flush=True)
        continue
    print(f'=== running {name} in its own subprocess ===', flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, 'inside_airbnb/_heldout_mlp_worker.py', name])
    if result.returncode != 0:
        raise SystemExit(f'{name} worker failed with exit code {result.returncode}')
    print(f'{name} finished in {time.time()-t0:.1f}s total (incl. process startup)', flush=True)

# ---- combine ----
oof_dev, pred_test, times, rows = {}, {}, {}, []
ydev = ytest = None
for name in MODELS:
    d = np.load(OUT / f'_heldout_mlp_part_{name}.npz')
    oof_dev[DISPLAY[name]] = d['dev_oof']
    pred_test[DISPLAY[name]] = d['test_pred']
    times[DISPLAY[name]] = float(d['seconds'])
    ydev, ytest = d['ydev'], d['ytest']
    rows.append((DISPLAY[name], mean_squared_error(ydev, d['dev_oof']), r2_score(ydev, d['dev_oof']),
                 mean_squared_error(ytest, d['test_pred']), r2_score(ytest, d['test_pred'])))

names = list(oof_dev)
Mdev = np.column_stack([oof_dev[n] for n in names])
Mtest = np.column_stack([pred_test[n] for n in names])
bm = lambda w: mean_squared_error(ydev, np.clip(Mdev @ w, 0, 92))
r = minimize(bm, np.full(len(names), 1 / len(names)), method='SLSQP', bounds=[(0, 1)] * len(names),
             constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-9, 'maxiter': 500})
w = r.x.copy(); w[w < 1e-4] = 0; w /= w.sum()
bl_dev = np.clip(Mdev @ w, 0, 92); bl_test = np.clip(Mtest @ w, 0, 92)
rows.append(('SLSQP blend', mean_squared_error(ydev, bl_dev), r2_score(ydev, bl_dev),
             mean_squared_error(ytest, bl_test), r2_score(ytest, bl_test)))

print()
for n, dm, dr, tm, tr in rows:
    print(f'{n:<18} devR2={dr:.3f}  TESTR2={tr:.3f}')
print('blend weights: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names, w)))

rep = ['HELD-OUT EVALUATION -- Setup C WITH MLP (7-mo history -> Feb-Mar-Apr 2026, clean target)',
       f'dev n={len(ydev)}, held-out test n={len(ytest)}, seed=42', '=' * 68,
       f'{"Model":<18}{"DevMSE":>9}{"DevR2":>8}{"TestMSE":>10}{"TestR2":>9}{"Time(s)":>10}']
for n, dm, dr, tm, tr in rows:
    tsec = times.get(n, float('nan'))
    rep.append(f'{n:<18}{dm:>9.2f}{dr:>8.3f}{tm:>10.2f}{tr:>9.3f}{tsec:>10.1f}')
rep.append('\nblend weights: ' + ', '.join(f'{n}={wi:.3f}' for n, wi in zip(names, w)))
txt = '\n'.join(rep)
(OUT / 'heldout_setupC_with_mlp_results.txt').write_text(txt, encoding='utf-8')
np.savez(OUT / 'oof_setupC_with_mlp_heldout.npz', ydev=ydev, ytest=ytest,
         **{f'dev_{n}': oof_dev[n] for n in names}, **{f'test_{n}': pred_test[n] for n in names},
         weights=w, names=np.array(names))
print('\nsaved outputs/heldout_setupC_with_mlp_results.txt')
