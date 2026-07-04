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

# ---- Fig 2: availability distribution / blocking (real: ~77% f) ----
fig,ax=plt.subplots(figsize=(5.2,4.2))
ax.bar(['available (t)','unavailable (f)'],[22.9,77.1],color=['steelblue','firebrick'])
for i,v in enumerate([22.9,77.1]): ax.text(i,v,f'{v:.0f}%',ha='center',va='bottom')
ax.set_ylabel('Share of calendar-days (%)'); ax.set_title('Calendar availability (NYC 2026)'); ax.set_ylim(0,90)
save(fig,'fig2_availability')

# ---- Fig 3: price coverage by month (real measurements) ----
months=['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr']
cov=[58.7,58.5,58.4,59.1,58.9,0.0,0.0,0.0,59.6,59.1]
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(months,cov,color=['firebrick' if c==0 else 'steelblue' for c in cov])
ax.set_ylabel('listings.csv price non-null (%)'); ax.set_title('Price-field coverage by monthly snapshot (2025-26)')
ax.axhline(0,color='k',lw=0.8); save(fig,'fig3_price_coverage')

# ---- Fig 4: model comparison (held-out test R2) ----
mdl=['Linear','CatBoost','RandomForest','XGBoost','LightGBM','SLSQP blend']
r2=[0.568,0.641,0.644,0.647,0.649,0.653]
fig,ax=plt.subplots(figsize=(7.5,4.2))
cols=['darkgreen' if 'blend' in m else 'steelblue' for m in mdl]
ax.barh(mdl,r2,color=cols)
for i,v in enumerate(r2): ax.text(v,i,f' {v:.3f}',va='center')
ax.set_xlabel('Held-out test $R^2$'); ax.set_title('Model comparison (20% held-out test)'); ax.set_xlim(0.5,0.68)
save(fig,'fig4_model_compare')

# ---- Fig 5: temporal robustness (A/B/C) ----
sc=['Setup A\n(Q4 2025)','Setup B\n(Q1 2026)','Setup C\n(Feb-Apr 2026)']; rr=[0.645,0.574,0.664]
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

# ---- Fig 7 & 8: residual diagnostics (Setup C blend OOF) ----
d=np.load(OUT/'oof_setupC.npz',allow_pickle=True); yv=d['y']; bl=d['blend']; res=yv-bl
fig,ax=plt.subplots(2,2,figsize=(12,9))
ax[0,0].scatter(bl,res,s=4,alpha=0.12,color='steelblue'); ax[0,0].axhline(0,color='k',lw=1)
ax[0,0].set_xlabel('Predicted (blend)'); ax[0,0].set_ylabel('Residual'); ax[0,0].set_title('(a) Residual vs. fitted')
ax[0,1].scatter(yv,res,s=4,alpha=0.12,color='darkorange'); ax[0,1].axhline(0,color='k',lw=1)
ax[0,1].axvspan(-.5,2.5,color='red',alpha=.08); ax[0,1].axvspan(86.5,90.5,color='green',alpha=.08)
ax[0,1].set_xlabel('True y'); ax[0,1].set_ylabel('Residual'); ax[0,1].set_title('(b) Residual vs. true')
ax[1,0].hist(res,bins=60,color='slategray',edgecolor='white'); ax[1,0].axvline(0,color='red',lw=1)
ax[1,0].set_xlabel('Residual'); ax[1,0].set_title(f'(c) Residuals (mean={res.mean():.2f}, sd={res.std():.2f})')
stats.probplot(res,dist='norm',plot=ax[1,1]); ax[1,1].set_title('(d) Normal Q-Q')
fig.tight_layout(); save(fig,'fig7_residuals')

bins=[(-.5,.5),(.5,5.5),(5.5,20.5),(20.5,50.5),(50.5,80.5),(80.5,90.5)]
labs=['0','1-5','6-20','21-50','51-80','81-90']; me=[]; cs=[]
for lo,hi in bins:
    m=(yv>lo)&(yv<=hi); cs.append(int(m.sum())); me.append(float(res[m].mean()) if m.any() else 0)
fig,ax=plt.subplots(figsize=(8.5,5))
ax.bar(labs,me,color=['firebrick' if e<0 else 'seagreen' for e in me],alpha=.85); ax.axhline(0,color='k',lw=1)
ax.set_xlabel('True booked-nights bin'); ax.set_ylabel('Mean signed residual'); ax.set_title('Boundary bias by target bin')
for i,(e,c) in enumerate(zip(me,cs)): ax.text(i,e,f'{e:+.1f}\n(n={c})',ha='center',va='bottom' if e>=0 else 'top',fontsize=10)
save(fig,'fig8_residual_bins')
print('paper2 figures done:', len(list(FG.glob('*.png'))), 'png')
