"""
target_setupC.py -- cleaner Feb-Mar-Apr 2026 demand target (Setup C).
Snapshots seeing Feb/Mar/Apr 2026 dates: Oct2025..Apr2026 (7). Clean rule:
booked if observed available >=once, last state 'f', seen in >=2 snapshots.
Universe = January 2026 listings (known at prediction time, end of Jan).
Outputs: target_setupC_clean.csv (y_clean, y_orig) + stats; caches q234_panel.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
P25 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb\Q4')
P26 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2026_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
SNAPS = [(1, P25/'October2025'), (2, P25/'November2025'), (3, P25/'December2025'),
         (4, P26/'January2026'), (5, P26/'February2026'), (6, P26/'March2026'), (7, P26/'April2026')]
TGT = ('2026-02', '2026-03', '2026-04'); CHUNK = 3_000_000

frames = []
for order, folder in SNAPS:
    for ch in pd.read_csv(folder/'calendar.csv', usecols=['listing_id', 'date', 'available'],
                          dtype={'listing_id':'int64','date':'str','available':'str'}, chunksize=CHUNK):
        ch = ch[ch['date'].str.slice(0, 7).isin(TGT)]
        if ch.empty: continue
        ch = ch.copy(); ch['order'] = np.int8(order); ch['av'] = (ch['available'] == 't')
        frames.append(ch[['listing_id', 'date', 'order', 'av']])
    print('read', folder.name, flush=True)
panel = pd.concat(frames, ignore_index=True).sort_values(['listing_id', 'date', 'order'])
panel.to_parquet(OUT/'q234_panel.parquet', index=False)
g = panel.groupby(['listing_id', 'date'])['av']
agg = pd.DataFrame({'n_obs': g.size(), 'ever_avail': g.max(), 'last_av': g.last(), 'first_av': g.first()})
agg['booked_clean'] = ((agg['n_obs'] >= 2) & (agg['ever_avail']) & (~agg['last_av'])).astype('int8')
agg['booked_orig'] = (agg['first_av'] & (~agg['last_av'])).astype('int8')
y = agg.groupby(level='listing_id')[['booked_clean', 'booked_orig']].sum().clip(0, 92)
univ = pd.read_csv(P26/'January2026'/'listings.csv', usecols=['id'])['id'].rename('listing_id').to_frame()
out = univ.merge(y, on='listing_id', how='left').fillna(0)
out['y_clean'] = out['booked_clean'].astype('int32'); out['y_orig'] = out['booked_orig'].astype('int32')
out[['listing_id', 'y_clean', 'y_orig']].to_csv(OUT/'target_setupC_clean.csv', index=False)
def st(n, v): return f'{n:<9} mean={v.mean():.2f} median={np.median(v):.1f} std={v.std():.2f} %zero={np.mean(v==0)*100:.1f}'
rep = 'SETUP C TARGET (Feb-Mar-Apr 2026, clean)\n'+'='*46+'\n'+st('y_clean', out['y_clean'].values)+'\n'+st('y_orig', out['y_orig'].values)
print('\n'+rep); (OUT/'target_setupC_clean_stats.txt').write_text(rep, encoding='utf-8')
