"""
target_q4_booking_diff.py
Demand target via snapshot differencing (isolates real bookings from host blocks).

For every Q4 2025 date (Oct 1 - Dec 31) and every listing, availability is tracked
across the six monthly snapshots that contain that date as a FUTURE date:
    Jul, Aug, Sep, Oct, Nov, Dec 2025  (scrape order 1..6).
A date counts as BOOKED if it was available ('t') in the EARLIEST snapshot that
saw it and unavailable ('f') in the LATEST snapshot that saw it -- i.e. it got
taken over the observation window. Dates already 'f' at first sight (host-blocked
from the start) are NOT counted. y = booked Q4 nights per listing, clipped [0,92].
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
SNAPS = [(1, BASE / 'Q3' / 'July2025'), (2, BASE / 'Q3' / 'August2025'),
         (3, BASE / 'Q3' / 'September2025'), (4, BASE / 'Q4' / 'October2025'),
         (5, BASE / 'Q4' / 'November2025'), (6, BASE / 'Q4' / 'December2025')]
Q4 = ('2025-10', '2025-11', '2025-12')
CHUNK = 3_000_000

frames = []
for order, folder in SNAPS:
    path = folder / 'calendar.csv'
    kept = 0
    for ch in pd.read_csv(path, usecols=['listing_id', 'date', 'available'],
                          dtype={'listing_id': 'int64', 'date': 'str', 'available': 'str'},
                          chunksize=CHUNK):
        ch = ch[ch['date'].str.slice(0, 7).isin(Q4)]
        if ch.empty:
            continue
        ch = ch.copy(); ch['order'] = np.int8(order)
        ch['av'] = (ch['available'] == 't')
        frames.append(ch[['listing_id', 'date', 'order', 'av']])
        kept += len(ch)
    print(f'  snap {order} {folder.name}: kept {kept:,} Q4 rows')

cal = pd.concat(frames, ignore_index=True)
del frames
print('total Q4 (listing,date,snapshot) rows:', f'{len(cal):,}')

cal = cal.sort_values(['listing_id', 'date', 'order'])
g = cal.groupby(['listing_id', 'date'])['av']
first_av = g.first()          # availability when first observed
last_av = g.last()            # availability when last observed
booked = (first_av & (~last_av))            # t -> f  == booked
y = booked.groupby(level='listing_id').sum().astype('int32').clip(0, 92)

# universe = October 2025 listings
oct_ids = pd.read_csv(BASE / 'Q4' / 'October2025' / 'listings.csv', usecols=['id'])['id']
out = (oct_ids.rename('listing_id').to_frame()
       .merge(y.rename('y_booked_q4'), on='listing_id', how='left').fillna(0))
out['y_booked_q4'] = out['y_booked_q4'].astype('int32')
out.to_csv(OUT / 'target_q4_2025_booking_diff.csv', index=False)

yv = out['y_booked_q4'].values
rep = '\n'.join([
    'TARGET (booking-differencing): newly-booked nights in Q4 2025  [0,92]',
    '=' * 64,
    f'listings (October 2025 universe) : {len(yv):,}',
    f'mean                             : {yv.mean():.2f}',
    f'median                           : {np.median(yv):.1f}',
    f'std                              : {yv.std():.2f}',
    f'skewness                         : {pd.Series(yv).skew():.2f}',
    f'share y == 0                     : {np.mean(yv == 0) * 100:.1f}%',
    f'share y >= 60                    : {np.mean(yv >= 60) * 100:.1f}%',
    f'max                              : {int(yv.max())}',
])
print('\n' + rep)
(OUT / 'target_q4_2025_booking_diff_stats.txt').write_text(rep, encoding='utf-8')

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(yv, bins=93, color='seagreen', edgecolor='none')
ax.set_xlabel('Newly-booked nights in Q4 2025 (y)'); ax.set_ylabel('Number of listings')
ax.set_title('Setup A demand target (booking-differencing, NYC Q4 2025)')
fig.tight_layout(); fig.savefig(OUT / 'target_q4_2025_booking_diff_hist.png', dpi=150)
print('\nsaved booking-diff target csv, stats, hist')
