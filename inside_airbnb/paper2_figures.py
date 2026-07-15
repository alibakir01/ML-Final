"""paper2_figures.py -- clean, high-DPI figures for the 2025-26 journal paper.
All from real data / saved OOF. Output: inside_airbnb/paper2/figures/*.png(+pdf)"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,
                     'xtick.labelsize':11,'ytick.labelsize':11})
OUT = Path('inside_airbnb/outputs'); FG = Path('inside_airbnb/paper2/figures')
def save(fig,n): fig.savefig(FG/f'{n}.png',dpi=300,bbox_inches='tight'); fig.savefig(FG/f'{n}.pdf',bbox_inches='tight'); plt.close(fig)

# ---- Fig 1: target distribution (Setup C, Feb-Mar-Apr 2026) ----
y = pd.read_csv(OUT/'target_setupC_clean.csv')['y_clean'].values
fig,ax=plt.subplots(figsize=(7,4.2))
ax.hist(y,bins=90,color='seagreen',edgecolor='none')
ax.set_xlabel('Booked nights in Feb-Mar-Apr 2026 (target)'); ax.set_ylabel('Number of listings')
ax.set_title('Target distribution (zero-inflated, bounded)')
save(fig,'fig1_target_dist')

# ---- Fig 2: availability distribution / blocking ----
# Recomputed live from the raw September 2025 calendar (same snapshot Table II is
# anchored to), so this number can never silently drift from the real data again.
# Reading only the 'available' column in chunks takes a few seconds.
_cal_path = Path('Internship/AirBnb_Inside/2025_Inside_Airbnb/Q3/September2025/calendar.csv')
_n_t = _n_f = 0
for _ch in pd.read_csv(_cal_path, usecols=['available'], chunksize=2_000_000):
    _vc = _ch['available'].value_counts()
    _n_t += _vc.get('t', 0); _n_f += _vc.get('f', 0)
_tot = _n_t + _n_f
avail_pct, unavail_pct = 100 * _n_t / _tot, 100 * _n_f / _tot
print(f'Fig 2 (Sep 2025 calendar, n={_tot:,}): available={avail_pct:.1f}% unavailable={unavail_pct:.1f}%')

fig,ax=plt.subplots(figsize=(5.2,4.2))
ax.bar(['available (t)','unavailable (f)'],[avail_pct,unavail_pct],color=['steelblue','firebrick'])
for i,v in enumerate([avail_pct,unavail_pct]): ax.text(i,v,f'{v:.0f}%',ha='center',va='bottom')
ax.set_ylabel('Share of calendar-days (%)'); ax.set_title('Calendar availability (NYC, Sep 2025 snapshot)'); ax.set_ylim(0,90)
save(fig,'fig2_availability')

# ---- Fig 3: price coverage by month (real measurements) ----
months=['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr']
cov=[58.7,58.5,58.4,59.1,58.9,0.0,0.0,0.0,59.6,59.1]
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(months,cov,color=['firebrick' if c==0 else 'steelblue' for c in cov])
ax.set_ylabel('listings.csv price non-null (%)'); ax.set_title('Price-field coverage by monthly snapshot (2025-26)')
ax.axhline(0,color='k',lw=0.8); save(fig,'fig3_price_coverage')

# ---- Fig 4: model comparison (held-out test R2), six learners incl. MLP ----
mdl=['Linear','MLP','CatBoost','RandomForest','XGBoost','LightGBM','SLSQP blend']
r2=[0.568,0.626,0.641,0.644,0.647,0.649,0.654]
fig,ax=plt.subplots(figsize=(7.5,4.6))
cols=['darkgreen' if 'blend' in m else 'steelblue' for m in mdl]
ax.barh(mdl,r2,color=cols)
for i,v in enumerate(r2): ax.text(v,i,f' {v:.3f}',va='center')
ax.set_xlabel('Held-out test $R^2$'); ax.set_title('Model comparison (20% held-out test)'); ax.set_xlim(0.5,0.68)
save(fig,'fig4_model_compare')

# ---- Fig 5: temporal robustness (A/B/C), six-learner blend incl. MLP in all three ----
sc=['Setup A\n(Q4 2025)','Setup B\n(Q1 2026)','Setup C\n(Feb-Apr 2026)']; rr=[0.645,0.575,0.665]
fig,ax=plt.subplots(figsize=(6.5,4.2))
ax.bar(sc,rr,color=['steelblue','steelblue','darkgreen'])
for i,v in enumerate(rr): ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
ax.set_ylabel('Blend $R^2$ (5-fold CV)'); ax.set_title('Cross-period robustness'); ax.set_ylim(0,0.75)
save(fig,'fig5_temporal')

# ---- Fig 6: ablation (levers) ----
lev=['Clean target','Deeper history','Tuning','Price feats','Review feats','Hurdle','Nbhd context']
val=[0.040,0.012,0.002,0.000,0.000,0.000,0.000]
fig,ax=plt.subplots(figsize=(7.5,4.2))
ax.barh(lev[::-1],val[::-1],color=['seagreen' if v>0.005 else 'grey' for v in val[::-1]])
for i,v in enumerate(val[::-1]): ax.text(v,i,f' +{v:.3f}',va='center')
ax.set_xlabel('$\\Delta R^2$ contribution'); ax.set_title('Ablation: only the target and history help')
save(fig,'fig6_ablation')

# ---- Fig 7: residual diagnostics (Setup C blend OOF) ----
# Sized for a double-column (full text-width, ~7in) placement in the manuscript --
# a single-column placement makes the 2x2 grid's tick/axis labels illegible.
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                     'xtick.labelsize':9.5,'ytick.labelsize':9.5})
# six-learner blend (adds the tuned MLP), recombined from cached per-model OOF -- see
# oof_setupC_blend6.npz for how it was derived (same cached predictions, no retraining)
d=np.load(OUT/'oof_setupC_blend6.npz',allow_pickle=True); yv=d['y']; bl=d['blend']; res=yv-bl
fig,ax=plt.subplots(2,2,figsize=(7.0,5.3))
ax[0,0].scatter(bl,res,s=4,alpha=0.12,color='steelblue'); ax[0,0].axhline(0,color='k',lw=1)
ax[0,0].set_xlabel('Predicted (blend)'); ax[0,0].set_ylabel('Residual'); ax[0,0].set_title('(a) Residual vs. fitted')
ax[0,1].scatter(yv,res,s=4,alpha=0.12,color='darkorange'); ax[0,1].axhline(0,color='k',lw=1)
ax[0,1].axvspan(-.5,2.5,color='red',alpha=.08); ax[0,1].axvspan(86.5,90.5,color='green',alpha=.08)
ax[0,1].set_xlabel('True y'); ax[0,1].set_ylabel('Residual'); ax[0,1].set_title('(b) Residual vs. true')
ax[1,0].hist(res,bins=60,color='slategray',edgecolor='white'); ax[1,0].axvline(0,color='red',lw=1)
ax[1,0].set_xlabel('Residual'); ax[1,0].set_title(f'(c) Residuals (mean={res.mean():.2f}, sd={res.std():.2f})')
stats.probplot(res,dist='norm',plot=ax[1,1]); ax[1,1].set_title('(d) Normal Q-Q')
ax[1,1].get_lines()[0].set_markersize(3)
fig.tight_layout(); save(fig,'fig7_residuals')

# ---- Fig 8: residual bins (single column placement is fine) ----
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,
                     'xtick.labelsize':11,'ytick.labelsize':11})
bins=[(-.5,.5),(.5,5.5),(5.5,20.5),(20.5,50.5),(50.5,80.5),(80.5,90.5)]
labs=['0','1-5','6-20','21-50','51-80','81-90']; me=[]; cs=[]
for lo,hi in bins:
    m=(yv>lo)&(yv<=hi); cs.append(int(m.sum())); me.append(float(res[m].mean()) if m.any() else 0)
fig,ax=plt.subplots(figsize=(8.5,5.4))
ax.bar(labs,me,color=['firebrick' if e<0 else 'seagreen' for e in me],alpha=.85); ax.axhline(0,color='k',lw=1)
ax.set_xlabel('True booked-nights bin'); ax.set_ylabel('Mean signed residual'); ax.set_title('Boundary bias by target bin')
ymin,ymax=min(me),max(me); pad=0.10*(ymax-ymin)
ax.set_ylim(ymin-pad-3.0, ymax+pad+3.0)  # headroom so labels below negative bars clear the x-tick text
for i,(e,c) in enumerate(zip(me,cs)):
    va='bottom' if e>=0 else 'top'; off=0.6 if e>=0 else -0.6
    ax.text(i,e+off,f'{e:+.1f}\n(n={c})',ha='center',va=va,fontsize=10)
save(fig,'fig8_residual_bins')

# ---- Fig 9: XGBoost gain importances (Setup C) ----
# Re-fit on the exact configuration behind the reported held-out numbers (features_setupC.parquet,
# target y_clean, 80/20 dev/test split seed=42, XGBoost params from heldout_setupC.py, fit on the
# full dev fold) so the bars come from a real, reproducible model rather than a redrawn old image.
# Sized for double-column placement, and with human-readable feature labels, so the 18 category
# names and their bars stay legible at print size.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

RS=42
dfC = pd.read_parquet(OUT/'features_setupC.parquet')
yC_full = dfC['y_clean'].astype(float).values
XC_full = dfC.drop(columns=['y_clean','listing_id']).reset_index(drop=True)
dev_idx, _test_idx = train_test_split(np.arange(len(yC_full)), test_size=0.20, random_state=RS)
XC = XC_full.iloc[dev_idx].reset_index(drop=True); yC = yC_full[dev_idx]
cat = XC.select_dtypes(exclude='number').columns.tolist()
card = {c: XC[c].astype(str).nunique() for c in cat}
cat_high=[c for c,n in card.items() if n>15]; cat_low=[c for c,n in card.items() if n<=15]
num = XC.select_dtypes(include='number').columns.tolist()

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

ppC = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num),
    ('low', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='m')),
                      ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_low),
    ('high', KFoldTE(cat_high), cat_high)])
XtC = ppC.fit_transform(XC, yC)
feat_names = list(ppC.named_transformers_['num'].get_feature_names_out(num)) \
    + list(ppC.named_transformers_['low'].named_steps['oh'].get_feature_names_out(cat_low)) + cat_high

xgb9 = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=3.0, random_state=RS, n_jobs=1,
                    tree_method='hist', importance_type='gain')
xgb9.fit(XtC, yC)
imp9 = xgb9.feature_importances_; imp9 = imp9/imp9.sum()

LABEL_MAP = {
    'hjan_booking_rate': "Jan. booking rate (most recent month)",
    'recency_booking_rate': "Recency-weighted booking rate",
    'hjan_booked': "Jan. booked nights",
    'estimated_occupancy_l365d': "Platform trailing occupancy (365d)",
    'h7_booked_total': "7-month booked total",
    'trend_jan_q4': "Trend: Jan. vs. Q4",
    'rev_recency_days': "Days since last review (reviews table)",
    'hq4_occ_rate': "Q4 occupancy rate",
    'hjan_occ': "Jan. occupied days",
    'hq3_occ_rate': "Q3 occupancy rate",
    'days_since_last_review': "Days since last review (listing field)",
    'host_listings_count': "Host listings count",
    'hq4_occupied': "Q4 occupied days",
    'host_is_superhost': "Host is superhost",
    'hq4_changes': "Q4 availability changes (churn)",
    'neighbourhood_group_cleansed_Manhattan': "Borough = Manhattan",
    'license_missing': "License field missing",
    'rev_365': "Reviews in trailing 365 days",
}
order9 = np.argsort(imp9)[::-1][:18]
names9 = [feat_names[i] for i in order9]; vals9 = [imp9[i] for i in order9]
labels9 = [LABEL_MAP.get(n, n) for n in names9]

plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                     'xtick.labelsize':10,'ytick.labelsize':9.5})
fig,ax=plt.subplots(figsize=(7.0,5.2))
ax.barh(labels9[::-1], vals9[::-1], color='seagreen')
ax.set_xlabel('XGBoost gain importance (normalised)')
ax.set_title('Top-18 features by XGBoost gain (Setup C)')
fig.tight_layout(); save(fig,'fig9_importance')

print('paper2 figures done:', len(list(FG.glob('*.png'))), 'png')
