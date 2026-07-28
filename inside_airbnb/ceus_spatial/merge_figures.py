"""
Produce two clean MERGED multi-panel figures for the condensed CEUS manuscript:
  fig_dynamics.pdf      = (a) city weekly booked nights, (b) borough trajectories   [old 15+16]
  fig_concentration.pdf = (a) unlicensed share by borough,  (b) demand by host size [old 17+18]
Vector PDF, no geopandas needed. Maps (old 11/12/13) are composed separately in LaTeX.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
BLUE="#08519c"; LBLUE="#9ecae1"
feat = pd.read_parquet(ROOT/"inside_airbnb/outputs/features_setupC.parquet",
        columns=["listing_id","neighbourhood_group_cleansed","license_missing","host_listings_count"])

# ---------------- Figure: intra-quarter dynamics (old 15 + 16) ----------------
p = pd.read_parquet(ROOT/"inside_airbnb/outputs/q1_panel.parquet"); p["date"]=pd.to_datetime(p["date"],errors="coerce")
ft = p[p.av==True ].groupby(["listing_id","date"])["order"].min()
lf = p[p.av==False].groupby(["listing_id","date"])["order"].max()
j  = pd.concat([ft.rename("t"), lf.rename("f")], axis=1)
bk = j[(j.t.notna()) & (j.f.notna()) & (j.f>j.t)].reset_index()[["listing_id","date"]]
bk = bk.merge(feat[["listing_id","neighbourhood_group_cleansed"]], on="listing_id", how="left")
bk["week"]=bk["date"].dt.to_period("W").apply(lambda r:r.start_time)
wk = bk.groupby("week").size().iloc[1:-1]
bt = bk.groupby([bk["week"],"neighbourhood_group_cleansed"]).size().unstack(fill_value=0).iloc[1:-1]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(wk.index, wk.values, marker="o", color=BLUE); ax[0].grid(alpha=0.3)
ax[0].set_ylabel("Booked nights (city total)"); ax[0].set_xlabel("Week (2026 Q1 window)")
ax[0].set_title("(a) City weekly demand")
for lab in ax[0].get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
for b in ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]:
    if b in bt: ax[1].plot(bt.index, bt[b], marker=".", label=b)
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3); ax[1].set_xlabel("Week"); ax[1].set_ylabel("Booked nights")
ax[1].set_title("(b) By borough")
for lab in ax[1].get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
fig.tight_layout(); fig.savefig(OUT/"fig_dynamics.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure: structural concentration (old 17 + 18) ----------------
df = feat.copy(); z = np.load(ROOT/"inside_airbnb/outputs/oof_setupC_blend6.npz")
df["y_pred"] = np.clip(z["blend"], 0, None)
BOROS=["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]
def dshare(sub, m):
    t=sub["y_pred"].sum(); return sub.loc[m,"y_pred"].sum()/t if t>0 else np.nan
rows=[]
for b in BOROS:
    s=df[df.neighbourhood_group_cleansed==b]
    rows.append((b,(s.license_missing==1).mean(), dshare(s, s.license_missing==1)))
comp=pd.DataFrame(rows,columns=["b","ls","ds"]).set_index("b").sort_values("ds")
hc=df["host_listings_count"].fillna(1).clip(lower=0)
df["tier"]=pd.cut(hc,[0,1,5,20,np.inf],labels=["1","2–5","6–20",">20"],include_lowest=True)
tier=df.groupby("tier",observed=True).agg(ls=("y_pred",lambda s:len(s)/len(df)),
        ds=("y_pred",lambda s:s.sum()/df["y_pred"].sum()))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
yb=np.arange(len(comp)); h=0.38
ax[0].barh(yb+h/2, comp.ls*100, height=h, color=LBLUE, label="listings")
ax[0].barh(yb-h/2, comp.ds*100, height=h, color=BLUE, label="recovered demand")
for i,(ls,ds) in enumerate(zip(comp.ls,comp.ds)):
    ax[0].text(ls*100+0.5,i+h/2,f"{ls*100:.0f}%",va="center",fontsize=7)
    ax[0].text(ds*100+0.5,i-h/2,f"{ds*100:.0f}%",va="center",fontsize=7,fontweight="bold")
ax[0].set_yticks(yb); ax[0].set_yticklabels(comp.index); ax[0].set_xlim(0,100)
ax[0].set_xlabel("% without displayed licence"); ax[0].set_title("(a) Compliance by borough")
ax[0].set_ylim(-0.7, len(comp)-0.3)
ax[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5,-0.42), ncol=2, frameon=False)
t=tier.reindex(["1","2–5","6–20",">20"])
xb=np.arange(len(t))
ax[1].bar(xb, t["ds"]*100, color=BLUE, width=0.6, label="demand")
ax[1].bar(xb, t["ls"]*100, color="none", edgecolor=BLUE, width=0.6, hatch="///", label="listings")
for i,d in enumerate(t["ds"]): ax[1].text(i,d*100+0.8,f"{d*100:.0f}%",ha="center",fontsize=7,fontweight="bold")
ax[1].set_xticks(xb); ax[1].set_xticklabels(t.index); ax[1].set_xlabel("Host portfolio size (listings)")
ax[1].set_ylabel("% "); ax[1].set_title("(b) Demand by operator size"); ax[1].legend(fontsize=7,frameon=False)
fig.tight_layout(); fig.savefig(OUT/"fig_concentration.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote fig_dynamics.pdf and fig_concentration.pdf")
