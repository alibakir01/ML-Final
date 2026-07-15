"""
target_occupancy_setupC.py -- TRUE naive occupancy (unavailable-day count) target for
Setup C (Feb-Mar-Apr 2026), built from the already-cached outputs/q234_panel.parquet
(listing_id, date, order, av) -- no need to re-scan the raw calendar.csv files.

Definition mirrors eda_target_q4_2025.py's Setup A "occupancy" proxy (read each date's
availability from the freshest/most relevant snapshot that observed it): for every
(listing_id, date) pair, take the LAST observed availability state across the snapshots
that saw that date (this is, in practice, that date's own month-of snapshot, since a
future date drops out of later calendars once it has passed). A date counts as occupied
(unavailable) when that last-observed state is 'f'. This is the naive occupancy count
that Table III / the ablation study compares against the differencing target (y_clean).

Universe = January 2026 listings (same as target_setupC.py / features_setupC.parquet).
Outputs: outputs/target_setupC_occupancy.csv (listing_id, y_occupancy)
         outputs/target_setupC_occupancy_stats.txt (mean/median/std/%zero/skew)
Run from repo root: python inside_airbnb/target_occupancy_setupC.py
"""
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path('inside_airbnb/outputs')

panel = pd.read_parquet(OUT / 'q234_panel.parquet')
panel = panel.sort_values(['listing_id', 'date', 'order'])
g = panel.groupby(['listing_id', 'date'])['av']
last_av = g.last()  # freshest observed availability for that date
occupied = (~last_av).astype('int8')  # 'f' = unavailable = occupied/blocked
y = occupied.groupby(level='listing_id').sum().clip(0, 92)

# universe = January 2026 listings, same as target_setupC.py
jan_ids = pd.read_csv(
    'Internship/AirBnb_Inside/2026_Inside_Airbnb/January2026/listings.csv',
    usecols=['id'])['id'].rename('listing_id').to_frame()
out = jan_ids.merge(y.rename('y_occupancy'), on='listing_id', how='left').fillna(0)
out['y_occupancy'] = out['y_occupancy'].astype('int32')
out.to_csv(OUT / 'target_setupC_occupancy.csv', index=False)

yv = out['y_occupancy'].values
rep = '\n'.join([
    'SETUP C NAIVE OCCUPANCY TARGET (Feb-Mar-Apr 2026, unavailable-day count)',
    '=' * 68,
    f'listings (January 2026 universe) : {len(yv):,}',
    f'mean                             : {yv.mean():.2f}',
    f'median                           : {np.median(yv):.1f}',
    f'std                              : {yv.std():.2f}',
    f'skewness                         : {pd.Series(yv).skew():.2f}',
    f'share y == 0                     : {np.mean(yv == 0) * 100:.1f}%',
    f'max                              : {int(yv.max())}',
])
print('\n' + rep)
(OUT / 'target_setupC_occupancy_stats.txt').write_text(rep, encoding='utf-8')
print('\nsaved target_setupC_occupancy.csv and _stats.txt')
