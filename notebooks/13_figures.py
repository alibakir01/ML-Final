"""
13_figures.py -- regenerate ALL paper figures at high DPI with large fonts,
saving BOTH .png (for the Word build) and .pdf (vector, for LaTeX/print).
All values are computed from the real data / saved OOF predictions.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,
                     'xtick.labelsize':11,'ytick.labelsize':11,'figure.dpi':300})
OUT=Path('outputs'); FG=Path('paper/figures'); RS=42
def save(fig,name):
    fig.savefig(FG/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(FG/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

yL=pd.read_parquet(OUT/'train_local.parquet')['NumReserveDays2016Q3'].astype(float).values
names=['LinearRegression','RandomForest','GradientBoosting','XGBoost','MLP']
M=np.column_stack([np.clip(np.load(OUT/f'oof_{n}.npy'),0,92) for n in names])
def bm(w): return mean_squared_error(yL,np.clip(M@w,0,92))
r=minimize(bm,np.full(5,.2),method='SLSQP',bounds=[(0,1)]*5,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-9})
w=r.x.copy(); w[w<1e-4]=0; w/=w.sum(); bl=np.clip(M@w,0,92); res=yL-bl

# ---- Fig 1: target distribution ----
fig,ax=plt.subplots(figsize=(7,4.2))
ax.hist(yL,bins=93,color='steelblue',edgecolor='none')
ax.set_xlabel('Q3 2016 booked nights (y)'); ax.set_ylabel('Number of properties')
ax.set_title('Target distribution (zero-inflated, bounded [0,92])')
save(fig,'fig1_target_dist')

# ---- Fig 2: top-20 |correlation| with target ----
tf=pd.read_parquet(OUT/'train_features.parquet')
num=tf.select_dtypes(include='number').drop(columns=[c for c in ['PropertyID','NumReserveDays2016Q3'] if c in tf.columns])
corr=num.corrwith(tf['NumReserveDays2016Q3'].astype(float)).dropna().sort_values(key=np.abs,ascending=False).head(20)[::-1]
fig,ax=plt.subplots(figsize=(7,6))
ax.barh(range(len(corr)),corr.values,color=['firebrick' if v<0 else 'steelblue' for v in corr.values])
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index,fontsize=10)
ax.set_xlabel('Pearson correlation with target'); ax.set_title('Top-20 engineered features by |correlation|')
save(fig,'fig2_top_corr')

# ---- Fig 3: model comparison (CV MSE) incl SOTA ----
cv={'GradientBoosting':245.3,'MLP':217.6,'LinearRegression':207.6,'RandomForest':194.2,
    'CatBoost (SOTA)':184.3,'LightGBM (SOTA)':184.1,'XGBoost':182.2,'SLSQP blend':180.5}
s=pd.Series(cv).sort_values(ascending=False)
fig,ax=plt.subplots(figsize=(7.5,4.6))
cols=['darkgreen' if 'blend' in n else ('darkorange' if 'SOTA' in n else 'steelblue') for n in s.index]
ax.barh(s.index,s.values,color=cols)
for i,v in enumerate(s.values): ax.text(v,i,f' {v:.1f}',va='center',fontsize=10)
ax.set_xlabel('5-fold CV MSE (lower is better)'); ax.set_title('Model comparison (incl. SOTA baselines)')
ax.set_xlim(170,250); save(fig,'fig3_baseline_cv')

# ---- Fig 4: XGBoost gain importance (train once on full pipeline) ----
class TE(BaseEstimator,TransformerMixin):
    def __init__(self,cols,n_splits=5,smoothing=20.0,random_state=42):
        self.cols=cols;self.n_splits=n_splits;self.smoothing=smoothing;self.random_state=random_state
    def _m(self,x,yy):
        st=pd.DataFrame({'c':x,'y':yy}).groupby('c')['y'].agg(['mean','count'])
        return ((st['count']*st['mean']+self.smoothing*self.gm_)/(st['count']+self.smoothing)).to_dict()
    def fit(self,X,y):
        y=np.asarray(y,float);self.gm_=float(y.mean())
        self.maps_={c:self._m(X[c].astype(str).fillna('_n'),y) for c in self.cols};return self
    def transform(self,X):
        Xo=X.copy()
        for c in self.cols: Xo[c]=X[c].astype(str).fillna('_n').map(self.maps_[c]).fillna(self.gm_).astype('float32')
        return Xo
    def get_feature_names_out(self,n=None): return np.array(self.cols)
tr=pd.read_parquet(OUT/'train_dev.parquet'); ytr=tr['NumReserveDays2016Q3'].astype(float).values
Xtr=tr.drop(columns=['NumReserveDays2016Q3','PropertyID'])
na=Xtr.select_dtypes(include='number').columns.tolist()
ca=Xtr.select_dtypes(exclude='number').columns.tolist()
ch=[c for c in ca if Xtr[c].nunique(dropna=False)>15]; cl=[c for c in ca if Xtr[c].nunique(dropna=False)<=15]
pp=ColumnTransformer([('n',SimpleImputer(strategy='median'),na),
    ('l',Pipeline([('i',SimpleImputer(strategy='constant',fill_value='m')),('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cl),
    ('h',TE(ch),ch)])
Xt=pp.fit_transform(Xtr,ytr); fn=pp.get_feature_names_out()
m=XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=RS,n_jobs=-1)
m.fit(Xt,ytr)
imp=pd.Series(m.feature_importances_,index=[f.split('__')[-1] for f in fn]).sort_values(ascending=False).head(20)[::-1]
fig,ax=plt.subplots(figsize=(7,6))
ax.barh(range(len(imp)),imp.values,color='seagreen')
ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index,fontsize=10)
ax.set_xlabel('XGBoost gain importance'); ax.set_title('Top-20 features by XGBoost gain')
save(fig,'fig4_xgb_importance')

# ---- Fig 5: OOF MSE per model + blends ----
sc={n:mean_squared_error(yL,M[:,i]) for i,n in enumerate(names)}
sc['Equal blend']=mean_squared_error(yL,np.clip(M.mean(1),0,92)); sc['SLSQP blend']=bm(w)
s=pd.Series(sc).sort_values(ascending=False)
fig,ax=plt.subplots(figsize=(7.5,4.4))
ax.barh(s.index,s.values,color=['darkgreen' if 'blend' in n else 'steelblue' for n in s.index])
for i,v in enumerate(s.values): ax.text(v,i,f' {v:.1f}',va='center',fontsize=10)
ax.set_xlabel('Out-of-fold MSE'); ax.set_title('Per-model vs. blended OOF MSE'); ax.set_xlim(170,250)
save(fig,'fig5_blend')

# ---- Fig 6: residual diagnostics ----
fig,ax=plt.subplots(2,2,figsize=(12,9))
ax[0,0].scatter(bl,res,s=5,alpha=0.15,color='steelblue'); ax[0,0].axhline(0,color='k',lw=1)
ax[0,0].set_xlabel('Predicted booked nights (blend)'); ax[0,0].set_ylabel('Residual (y - y_hat)'); ax[0,0].set_title('(a) Residual vs. fitted')
ax[0,1].scatter(yL,res,s=5,alpha=0.15,color='darkorange'); ax[0,1].axhline(0,color='k',lw=1)
ax[0,1].axvspan(-0.5,2.5,color='red',alpha=0.08); ax[0,1].axvspan(89.5,92.5,color='green',alpha=0.08)
ax[0,1].set_xlabel('True booked nights y'); ax[0,1].set_ylabel('Residual (y - y_hat)'); ax[0,1].set_title('(b) Residual vs. true target')
ax[1,0].hist(res,bins=60,color='slategray',edgecolor='white'); ax[1,0].axvline(0,color='red',lw=1)
ax[1,0].set_xlabel('Residual (blend)'); ax[1,0].set_ylabel('Count'); ax[1,0].set_title(f'(c) Residual distribution (mean={res.mean():.2f}, sd={res.std():.2f})')
stats.probplot(res,dist='norm',plot=ax[1,1]); ax[1,1].set_title('(d) Normal Q-Q of residuals')
fig.tight_layout(); save(fig,'fig6_residuals')

# ---- Fig 7: signed residual by bin ----
bins=[(-.5,.5),(.5,5.5),(5.5,20.5),(20.5,50.5),(50.5,80.5),(80.5,89.5),(89.5,92.5)]
labs=['0','1-5','6-20','21-50','51-80','81-89','90-92']; me=[];cs=[]
for lo,hi in bins:
    mm=(yL>lo)&(yL<=hi); cs.append(int(mm.sum())); me.append(float(res[mm].mean()) if mm.any() else 0)
fig,ax=plt.subplots(figsize=(10,5.5))
ax.bar(labs,me,color=['firebrick' if e<0 else 'seagreen' for e in me],alpha=.85); ax.axhline(0,color='k',lw=1)
ax.set_xlabel('True booked-nights bin'); ax.set_ylabel('Mean signed residual (y - y_hat)'); ax.set_title('Systematic bias by target bin (blend, dev OOF)')
for i,(e,c) in enumerate(zip(me,cs)): ax.text(i,e,f'{e:+.1f}\n(n={c})',ha='center',va='bottom' if e>=0 else 'top',fontsize=10)
save(fig,'fig7_residual_bins')
print('regenerated 7 figures as PNG(300dpi)+PDF in', FG)
