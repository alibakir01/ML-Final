"""
target_q1_clean.py -- cleaner Q1 2026 demand target (less host-block / observation noise).
Rule: a Q1 date counts as BOOKED only if (i) it was observed AVAILABLE ('t') in at
least one snapshot, (ii) its LAST observed state is unavailable ('f'), and (iii) it
was observed in >= 2 snapshots (so a genuine availability->booked transition is
observable). Dates blocked from the start or seen only once are excluded -> less noise.
Also caches the per-(listing,date) panel so future target variants need no re-read.
Outputs: target_q1_2026_clean.csv, target_q1_2026_clean_stats.txt, q1_panel.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
P25 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb\Q4')
P26 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2026_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
SNAPS = [(1, P25/'October2025'), (2, P25/'November2025'), (3, P25/'December2025'),
         (4, P26/'January2026'), (5, P26/'February2026'), (6, P26/'March2026')]
Q1 = ('2026-01', '2026-02', '2026-03'); CHUNK = 3_000_000

frames = []
for order, folder in SNAPS:
    for ch in pd.read_csv(folder/'calendar.csv', usecols=['listing_id', 'date', 'available'],
                          dtype={'listing_id':'int64','date':'str','available':'str'}, chunksize=CHUNK):
        ch = ch[ch['date'].str.slice(0, 7).isin(Q1)]
        if ch.empty: continue
        ch = ch.copy(); ch['order'] = np.int8(order); ch['av'] = (ch['available'] == 't')
        frames.append(ch[['listing_id', 'date', 'order', 'av']])
    print('read', folder.name, flush=True)
panel = pd.concat(frames, ignore_index=True).sort_values(['listing_id', 'date', 'order'])
panel.to_parquet(OUT/'q1_panel.parquet', index=False)

g = panel.groupby(['listing_id', 'date'])['av']
agg = pd.DataFrame({'n_obs': g.size(), 'ever_avail': g.max(), 'last_av': g.last()})
agg['booked_clean'] = ((agg['n_obs'] >= 2) & (agg['ever_avail']) & (~agg['last_av'])).astype('int8')
agg['booked_orig'] = (g.first() & (~g.last())).astype('int8')   # original definition
y = agg.groupby(level='listing_id')[['booked_clean', 'booked_orig']].sum().clip(0, 92)

univ = pd.read_csv(P26/'January2026'/'listings.csv', usecols=['id'])['id'].rename('listing_id').to_frame()
out = univ.merge(y, on='listing_id', how='left').fillna(0)
out['y_clean'] = out['booked_clean'].astype('int32'); out['y_orig'] = out['booked_orig'].astype('int32')
out[['listing_id', 'y_clean', 'y_orig']].to_csv(OUT/'target_q1_2026_clean.csv', index=False)

def st(name, v):
    return f'{name:<10} mean={v.mean():.2f} median={np.median(v):.1f} std={v.std():.2f} %zero={np.mean(v==0)*100:.1f} corr_w_orig={np.corrcoef(v, out["y_orig"])[0,1]:.3f}'
rep = 'CLEANER Q1 2026 TARGET\n' + '='*50 + '\n' + st('y_clean', out['y_clean'].values) + '\n' + st('y_orig', out['y_orig'].values)
print('\n' + rep)
(OUT/'target_q1_2026_clean_stats.txt').write_text(rep, encoding='utf-8')
