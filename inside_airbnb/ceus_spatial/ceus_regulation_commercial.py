"""
CEUS "so-what" layer: regulation compliance + commercial concentration of the
recovered STR demand. Turns the (deliberately) weak spatial-clustering result
into a substantive urban / policy finding, using ONLY existing project outputs.

INPUTS (relative to project root ML-Final/):
  inside_airbnb/outputs/oof_setupC_blend6.npz    # blend OOF preds: y (true), blend (pred)
  inside_airbnb/outputs/features_setupC.parquet  # license_missing, host_*_listings_count, borough

OUTPUTS:
  inside_airbnb/ceus_spatial/figures/fig17_compliance.pdf   (vector)
  inside_airbnb/ceus_spatial/figures/fig18_commercial.pdf   (vector)
  prints every number used in the manuscript text.

Demand measure = blend out-of-fold PREDICTED quarterly booked nights, clipped at 0
(consistent with Section 5.9). Observed (y_true) values are printed alongside as a
robustness check; they match the predicted shares closely.

NOTE: `license_missing` is a PROXY. Inside Airbnb exposes a `license` field; a missing
value means no registration number is *displayed*, not a legal determination of
illegality (some listings may be exempt or registered off-platform). Phrase accordingly.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)

z = np.load(ROOT/"inside_airbnb/outputs/oof_setupC_blend6.npz", allow_pickle=True)
feat = pd.read_parquet(ROOT/"inside_airbnb/outputs/features_setupC.parquet",
        columns=["listing_id","neighbourhood_group_cleansed","license_missing",
                 "host_listings_count","host_total_listings_count"])
df = feat.copy()
df["y_true"] = z["y"]
df["y_pred"] = np.clip(z["blend"], 0, None)          # demand can't be negative
assert len(df) == len(z["y"]), "row misalignment"

BOROS = ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]

def share(mask, col="y_pred", frame=df):
    tot = frame[col].sum()
    return frame.loc[mask, col].sum()/tot if tot > 0 else np.nan

# ================================================================ 1) COMPLIANCE
unl = df["license_missing"] == 1
print("=== REGULATION / LICENSE COMPLIANCE ===")
print(f"Listings without displayed licence : {unl.mean():6.1%}  ({int(unl.sum())}/{len(df)})")
print(f"Share of PREDICTED demand unlicensed: {share(unl):6.1%}")
print(f"Share of OBSERVED  demand unlicensed: {share(unl, 'y_true'):6.1%}")

rows = []
for b in BOROS:
    sub = df[df.neighbourhood_group_cleansed == b]
    rows.append(dict(borough=b,
                     list_share=(sub.license_missing==1).mean(),
                     dem_share=share(sub.license_missing==1, frame=sub),
                     n=len(sub)))
comp = pd.DataFrame(rows).set_index("borough")
print("\nBy borough  (listing-share vs demand-share unlicensed):")
print((comp[["list_share","dem_share"]]*100).round(1))

# ---- Figure 17: grouped horizontal bars by borough
order = comp.sort_values("dem_share").index.tolist()
c = comp.loc[order]
y = np.arange(len(order)); h = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(y+h/2, c.list_share*100, height=h, color="#9ecae1", label="Share of listings unlicensed")
ax.barh(y-h/2, c.dem_share*100, height=h, color="#08519c", label="Share of recovered demand unlicensed")
for i,(ls,ds) in enumerate(zip(c.list_share, c.dem_share)):
    ax.text(ls*100+0.5, i+h/2, f"{ls*100:.0f}%", va="center", fontsize=8)
    ax.text(ds*100+0.5, i-h/2, f"{ds*100:.0f}%", va="center", fontsize=8, fontweight="bold")
ax.axvline(unl.mean()*100, color="grey", ls="--", lw=0.8)
ax.text(unl.mean()*100-1, -0.72, f"city avg {unl.mean()*100:.0f}%", color="grey",
        fontsize=8, ha="right")
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlabel("Percent of listings / recovered demand without a displayed licence")
ax.set_xlim(0, 100); ax.set_ylim(-1.1, len(order)-0.2)
ax.set_title("STR demand concentrates in listings without a displayed registration", pad=10)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT/"fig17_compliance.pdf", bbox_inches="tight"); plt.close(fig)

# ================================================================ 2) COMMERCIAL CONCENTRATION
hc = df["host_listings_count"].fillna(1).clip(lower=0)
bins   = [0, 1, 5, 20, np.inf]
labels = ["Single listing (1)", "Small (2–5)", "Mid (6–20)", "Commercial (>20)"]
df["host_tier"] = pd.cut(hc, bins=bins, labels=labels, include_lowest=True)
tier = df.groupby("host_tier", observed=True).agg(
        list_share=("y_pred", lambda s: len(s)/len(df)),
        dem_pred =("y_pred", lambda s: s.sum()/df["y_pred"].sum()),
        dem_true =("y_true", lambda s: s.sum()/df["y_true"].sum()))
print("\n=== COMMERCIAL CONCENTRATION (by host portfolio size) ===")
print((tier*100).round(1))
print(f"Demand from multi-listing hosts (>=2): {share(hc>=2):.1%} of predicted "
      f"({(hc>=2).mean():.1%} of listings)")
print(f"Demand from commercial hosts   (>20): {share(hc>20):.1%} of predicted "
      f"({(hc>20).mean():.1%} of listings)")

# Lorenz-style concentration curve: order listings by host size, cumulative demand vs listings
o = df.sort_values("host_listings_count", ascending=True)
cum_list = np.arange(1, len(o)+1)/len(o)
cum_dem  = o["y_pred"].cumsum().values / o["y_pred"].sum()
# concentration ratio at the top decile of operators
top10_cut = o["host_listings_count"].quantile(0.90)
top10_dem = share(df.host_listings_count >= top10_cut)
print(f"Top 10% largest operators (>= {top10_cut:.0f} listings) hold "
      f"{top10_dem:.1%} of predicted demand")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
# (a) demand share by tier
t = tier.reindex(labels)
ax[0].bar(range(len(t)), t["dem_pred"]*100, color="#08519c", width=0.6, label="Share of demand")
ax[0].bar(range(len(t)), t["list_share"]*100, color="none", edgecolor="#08519c",
          width=0.6, hatch="///", label="Share of listings")
for i,(d,l) in enumerate(zip(t["dem_pred"], t["list_share"])):
    ax[0].text(i, d*100+0.8, f"{d*100:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax[0].set_xticks(range(len(t))); ax[0].set_xticklabels(t.index, rotation=20, ha="right", fontsize=8)
ax[0].set_ylabel("Percent"); ax[0].set_title("(a) Recovered demand by host portfolio size")
ax[0].legend(fontsize=8, frameon=False)
# (b) concentration curve
ax[1].plot(cum_list*100, cum_dem*100, color="#08519c", lw=2)
ax[1].plot([0,100],[0,100], color="grey", ls="--", lw=0.8, label="equal share")
ax[1].fill_between(cum_list*100, cum_dem*100, cum_list*100, color="#9ecae1", alpha=0.4)
ax[1].set_xlabel("Cumulative % of listings (smallest → largest operator)")
ax[1].set_ylabel("Cumulative % of recovered demand")
ax[1].set_title("(b) Demand concentration by operator size")
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT/"fig18_commercial.pdf", bbox_inches="tight"); plt.close(fig)

print(f"\nFigures written to: {OUT}")
