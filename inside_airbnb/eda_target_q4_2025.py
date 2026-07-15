"""
eda_target_q4_2025.py
Setup A target construction + EDA for the Inside Airbnb replication.

Target = number of OCCUPIED (booked/blocked) nights per listing in Q4 2025
(Oct+Nov+Dec). Each month's occupancy is read from THAT month's own snapshot
calendar (the closest, most-settled forward view), so:
    Oct 2025 occupied  <- 2025/Q4/October2025/calendar.csv,  dates 2025-10-*
    Nov 2025 occupied  <- 2025/Q4/November2025/calendar.csv, dates 2025-11-*
    Dec 2025 occupied  <- 2025/Q4/December2025/calendar.csv, dates 2025-12-*
An occupied night is a calendar row with available == 'f'.
NOTE: Inside Airbnb 'available=f' conflates guest bookings with host blocks;
this is the standard occupancy proxy and is flagged as a limitation.

Outputs:
  inside_airbnb/outputs/target_q4_2025.csv      (listing_id, y, per-month parts)
  inside_airbnb/outputs/target_q4_2025_stats.txt
  inside_airbnb/outputs/target_q4_2025_hist.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb\Q4')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
MONTHS = [('October2025', '2025-10'), ('November2025', '2025-11'), ('December2025', '2025-12')]
CHUNK = 3_000_000


def occupied_for_month(folder, ym):
    """Return Series: listing_id -> #occupied days within month `ym`."""
    path = ROOT / folder / 'calendar.csv'
    occ = None
    obs = None
    rows = 0
    for ch in pd.read_csv(path, usecols=['listing_id', 'date', 'available'],
                          dtype={'listing_id': 'int64', 'date': 'str', 'available': 'str'},
                          chunksize=CHUNK):
        ch = ch[ch['date'].str.slice(0, 7) == ym]
        rows += len(ch)
        if ch.empty:
            continue
        ch['occ'] = (ch['available'] == 'f').astype('int32')
        g = ch.groupby('listing_id')['occ'].agg(['sum', 'count'])
        occ = g['sum'] if occ is None else occ.add(g['sum'], fill_value=0)
        obs = g['count'] if obs is None else obs.add(g['count'], fill_value=0)
    print(f'  {folder}: {ym} rows={rows:,}  listings={0 if occ is None else len(occ):,}')
    return (occ.rename(f'occ_{ym[-2:]}').astype('int32'),
            obs.rename(f'days_{ym[-2:]}').astype('int32'))

print('Building Q4 2025 occupancy target from month-of snapshots...')
parts = []
for folder, ym in MONTHS:
    occ, obs = occupied_for_month(folder, ym)
    parts.append(occ); parts.append(obs)

df = pd.concat(parts, axis=1).fillna(0).astype('int32')
df['y_occupied_q4'] = (df[['occ_10', 'occ_11', 'occ_12']].sum(axis=1)).clip(0, 92).astype('int32')
df['days_observed'] = df[['days_10', 'days_11', 'days_12']].sum(axis=1)
df = df.reset_index().rename(columns={'index': 'listing_id'})

# universe = listings present in the start-of-quarter (October) snapshot
oct_ids = pd.read_csv(ROOT / 'October2025' / 'listings.csv', usecols=['id'])['id']
df = df.merge(oct_ids.rename('listing_id'), on='listing_id', how='right').fillna(0)
df['y_occupied_q4'] = df['y_occupied_q4'].astype('int32')

df.to_csv(OUT / 'target_q4_2025.csv', index=False)

y = df['y_occupied_q4'].values
stats = [
    'TARGET: occupied (booked/blocked) nights in Q4 2025  [0, 92]',
    '=' * 62,
    f'listings (October 2025 universe) : {len(y):,}',
    f'mean                             : {y.mean():.2f}',
    f'median                           : {np.median(y):.1f}',
    f'std                              : {y.std():.2f}',
    f'skewness                         : {pd.Series(y).skew():.2f}',
    f'share y == 0                     : {np.mean(y == 0) * 100:.1f}%',
    f'share y >= 90                    : {np.mean(y >= 90) * 100:.1f}%',
    f'share y == 92                    : {np.mean(y == 92) * 100:.1f}%',
    f'max                              : {int(y.max())}',
]
rep = '\n'.join(stats)
print('\n' + rep)
(OUT / 'target_q4_2025_stats.txt').write_text(rep, encoding='utf-8')

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(y, bins=93, color='steelblue', edgecolor='none')
ax.set_xlabel('Occupied nights in Q4 2025 (y)'); ax.set_ylabel('Number of listings')
ax.set_title('Setup A target distribution (Inside Airbnb, NYC, Q4 2025)')
fig.tight_layout(); fig.savefig(OUT / 'target_q4_2025_hist.png', dpi=150)
print('\nsaved target_q4_2025.csv, _stats.txt, _hist.png')
