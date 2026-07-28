"""
CEUS spatial + temporal layer for the STR-demand paper.

Reproduces the 6 urban figures (Figure 11-16) used in the CEUS version of the
manuscript, straight from our own project outputs. No new data needed.

INPUTS (paths relative to project root ML-Final/):
  inside_airbnb/outputs/oof_setupC_blend6.npz      # blend OOF predictions: y, blend
  inside_airbnb/outputs/features_setupC.parquet    # listing_id, neighbourhood_cleansed,
                                                   # neighbourhood_group_cleansed, lat, lon
  inside_airbnb/outputs/q1_panel.parquet           # listing_id, date, order, av (raw calendars)
  Internship/AirBnb_Inside/2026_Inside_Airbnb/January2026/neighbourhoods.geojson

OUTPUTS:
  inside_airbnb/ceus_spatial/figures/Figure_11..16.pdf  (vector)
  prints Global Moran's I, borough means/shares, peak week.

NOTE ON DATA BASIS (important for the captions):
  Figures 11-14  = MODEL PREDICTIONS (blend OOF y_pred).  -> "the demand surface the model produces"
  Figures 15-16  = OBSERVED demand reconstructed from the raw calendars (differencing),
                   because predictions are quarterly only (no daily resolution).

RUN:
  pip install pyarrow geopandas libpysal esda mapclassify matplotlib
  python inside_airbnb/ceus_spatial/ceus_spatial_temporal.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local

ROOT = Path(__file__).resolve().parents[2]          # ML-Final/
OUT  = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
GEOJSON = ROOT / "Internship/AirBnb_Inside/2026_Inside_Airbnb/January2026/neighbourhoods.geojson"

# ---------------------------------------------------------------- load
z    = np.load(ROOT/"inside_airbnb/outputs/oof_setupC_blend6.npz", allow_pickle=True)
feat = pd.read_parquet(ROOT/"inside_airbnb/outputs/features_setupC.parquet",
        columns=["listing_id","neighbourhood_cleansed","neighbourhood_group_cleansed",
                 "latitude","longitude"])
df = feat.copy(); df["y_true"]=z["y"]; df["y_pred"]=z["blend"]; df["residual"]=df["y_true"]-df["y_pred"]
assert len(df)==len(z["y"]), "row misalignment"

# ---------------------------------------------------------------- spatial (predictions)
nbhd = (df.groupby("neighbourhood_cleansed")
          .agg(mean_pred=("y_pred","mean"), mean_resid=("residual","mean"), n=("y_pred","size"))
          .reset_index())
gdf = gpd.read_file(GEOJSON).merge(nbhd, left_on="neighbourhood",
                                   right_on="neighbourhood_cleansed", how="left")

def choro(col, title, fname, cmap):
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(7,7))
    gdf.plot(column=col, cmap=cmap, linewidth=0.2, edgecolor="grey",
             missing_kwds={"color":"lightgrey","label":"no data"}, ax=ax)
    # build the colorbar manually and keep it vector (no small raster strip)
    sm = mpl.cm.ScalarMappable(cmap=cmap,
         norm=mpl.colors.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max()))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.solids.set_rasterized(False)
    ax.set_title(title); ax.axis("off")
    fig.savefig(OUT/fname, bbox_inches="tight"); plt.close(fig)

choro("mean_pred","Predicted quarterly booked nights by neighbourhood (NYC)","Figure_11_demand_map.pdf","viridis")
choro("mean_resid","Mean prediction residual by neighbourhood (true - predicted)","Figure_12_residual_map.pdf","RdBu")

g = gdf.dropna(subset=["mean_pred"]).reset_index(drop=True)
w = Queen.from_dataframe(g, use_index=False); w.transform="r"
mi = Moran(g["mean_pred"].values, w)
lisa = Moran_Local(g["mean_pred"].values, w)
lab={1:"High-High (hot spot)",2:"Low-High",3:"Low-Low (cold spot)",4:"High-Low"}
g["lisa"]=[lab[q] if p<0.05 else "not significant" for q,p in zip(lisa.q, lisa.p_sim)]
cols={"High-High (hot spot)":"#d7191c","Low-Low (cold spot)":"#2c7bb6","Low-High":"#abd9e9",
      "High-Low":"#fdae61","not significant":"lightgrey"}
fig,ax=plt.subplots(figsize=(7,7))
for l,sub in g.groupby("lisa"): sub.plot(ax=ax,color=cols[l],edgecolor="grey",linewidth=0.2,label=l)
ax.legend(loc="upper left",fontsize=8,title="LISA cluster"); ax.set_title("Local clusters of predicted demand (LISA, p<0.05)"); ax.axis("off")
fig.savefig(OUT/"Figure_13_LISA.pdf",bbox_inches="tight"); plt.close(fig)

boro=(df.groupby("neighbourhood_group_cleansed").agg(mean_pred=("y_pred","mean"),n=("y_pred","size")).sort_values("mean_pred"))
fig,ax=plt.subplots(figsize=(6,4))
ax.barh(boro.index,boro["mean_pred"],color="#2c7bb6")
for i,v in enumerate(boro["mean_pred"]): ax.text(v,i,f" {v:.1f}",va="center")
ax.set_xlabel("Mean predicted booked nights (quarter)"); ax.set_title("Predicted STR demand by NYC borough")
fig.savefig(OUT/"Figure_14_borough.pdf",bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- temporal (observed, from raw calendars)
p = pd.read_parquet(ROOT/"inside_airbnb/outputs/q1_panel.parquet"); p["date"]=pd.to_datetime(p["date"],errors="coerce")
ft = p[p.av==True ].groupby(["listing_id","date"])["order"].min()
lf = p[p.av==False].groupby(["listing_id","date"])["order"].max()
j  = pd.concat([ft.rename("t"), lf.rename("f")], axis=1)
bk = j[(j.t.notna()) & (j.f.notna()) & (j.f>j.t)].reset_index()[["listing_id","date"]]   # booked = available->unavailable
bk = bk.merge(feat[["listing_id","neighbourhood_group_cleansed"]], on="listing_id", how="left")
bk["week"]=bk["date"].dt.to_period("W").apply(lambda r:r.start_time)
wk = bk.groupby("week").size().iloc[1:-1]                        # trim partial weeks

fig,ax=plt.subplots(figsize=(7,4))
ax.plot(wk.index,wk.values,marker="o",color="#2c7bb6"); ax.grid(alpha=0.3); fig.autofmt_xdate()
ax.set_ylabel("Booked nights (city total)"); ax.set_xlabel("Week (2026 Q1 window)")
ax.set_title("Recovered intra-quarter demand trajectory (NYC)")
fig.savefig(OUT/"Figure_15_demand_over_time.pdf",bbox_inches="tight"); plt.close(fig)

bt = bk.groupby([bk["week"],"neighbourhood_group_cleansed"]).size().unstack(fill_value=0).iloc[1:-1]
fig,ax=plt.subplots(figsize=(7,4))
for b in ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]:
    if b in bt: ax.plot(bt.index,bt[b],marker=".",label=b)
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.autofmt_xdate()
ax.set_ylabel("Booked nights"); ax.set_xlabel("Week"); ax.set_title("Demand trajectory by borough (spatial mix stays stable)")
fig.savefig(OUT/"Figure_16_borough_time.pdf",bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- report
print("=== CEUS spatial/temporal results ===")
print(f"Global Moran's I (neighbourhood mean pred) = {mi.I:.3f}  p = {mi.p_sim:.4f}  (n={len(g)})")
print("Borough mean predicted booked nights:")
for b,r in boro.iloc[::-1].iterrows(): print(f"   {b}: {r['mean_pred']:.1f} (n={int(r['n'])})")
mon = bk.groupby([bk['date'].dt.to_period('M').astype(str),'neighbourhood_group_cleansed']).size().unstack(fill_value=0)
print("\nBorough SHARE of demand by month (stable => no space-time interaction):")
print(mon.div(mon.sum(axis=1),axis=0).round(3))
print(f"\nPeak week: {wk.idxmax().date()}  ({int(wk.max())} booked nights)")
print(f"Figures written to: {OUT}")
