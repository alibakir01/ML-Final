"""
target_q4_clean.py -- cleaner Q4 2025 demand target (same rule as the Q1 cleaner).
A Q4 date counts BOOKED if observed available ('t') at least once, last observed
state is 'f', and it was seen in >=2 snapshots. Snapshots seeing Q4 dates:
Jul..Dec 2025. Universe = October 2025 listings.
Outputs: target_q4_2025_clean.csv (y_clean, y_orig) + stats.
"""
from pathlib import Path
import numpy as np, pandas as pd
B = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
SNAPS = [(1, B/'Q3'/'July2025'), (2, B/'Q3'/'August2025'), (3, B/'Q3'/'September2025'),
         (4, B/'Q4'/'October2025'), (5, B/'Q4'/'November2025'), (6, B/'Q4'/'December2025')]
Q4 = ('2025-10', '2025-11', '2025-12'); CHUNK = 3_000_000

frames = []
for order, folder in SNAPS:
    for ch in pd.read_csv(folder/'calendar.csv', usecols=['listing_id', 'date', 'available'],
                          dtype={'listing_id':'int64','date':'str','available':'str'}, chunksize=CHUNK):
        ch = ch[ch['date'].str.slice(0, 7).isin(Q4)]
        if ch.empty: continue
        ch = ch.copy(); ch['order'] = np.int8(order); ch['av'] = (ch['available'] == 't')
        frames.append(ch[['listing_id', 'date', 'order', 'av']])
    print('read', folder.name, flush=True)
panel = pd.concat(frames, ignore_index=True).sort_values(['listing_id', 'date', 'order'])
g = panel.groupby(['listing_id', 'date'])['av']
agg = pd.DataFrame({'n_obs': g.size(), 'ever_avail': g.max(), 'last_av': g.last(),
                    'first_av': g.first()})
agg['booked_clean'] = ((agg['n_obs'] >= 2) & (agg['ever_avail']) & (~agg['last_av'])).astype('int8')
agg['booked_orig'] = (agg['first_av'] & (~agg['last_av'])).astype('int8')
y = agg.groupby(level='listing_id')[['booked_clean', 'booked_orig']].sum().clip(0, 92)
univ = pd.read_csv(B/'Q4'/'October2025'/'listings.csv', usecols=['id'])['id'].rename('listing_id').to_frame()
out = univ.merge(y, on='listing_id', how='left').fillna(0)
out['y_clean'] = out['booked_clean'].astype('int32'); out['y_orig'] = out['booked_orig'].astype('int32')
out[['listing_id', 'y_clean', 'y_orig']].to_csv(OUT/'target_q4_2025_clean.csv', index=False)
def st(n, v): return f'{n:<9} mean={v.mean():.2f} median={np.median(v):.1f} std={v.std():.2f} %zero={np.mean(v==0)*100:.1f}'
rep = 'CLEANER Q4 2025 TARGET\n'+'='*44+'\n'+st('y_clean', out['y_clean'].values)+'\n'+st('y_orig', out['y_orig'].values)
print('\n'+rep); (OUT/'target_q4_2025_clean_stats.txt').write_text(rep, encoding='utf-8')
