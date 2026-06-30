"""
prep_setupB.py (v2) -- Setup B: predict Q1 2026 booked nights from Q4 2025 behaviour.
Target = newly-booked Q1 2026 nights (differencing over Oct2025..Mar2026, dates 2026-01/02/03).
Features = Q4 2025 (Oct/Nov/Dec) calendar behaviour + Dec-2025 listings static + price
dynamics from Oct/Nov/Dec listings (calendar price is empty, so listings price is used).
Universe = January-2026 listings. All features dated <= 2025-12-31 (no leakage).
Output: inside_airbnb/outputs/features_setupB.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

P25 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2025_Inside_Airbnb\Q4')
P26 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2026_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
SNAPS = [(1, P25/'October2025', True, '10'), (2, P25/'November2025', True, '11'), (3, P25/'December2025', True, '12'),
         (4, P26/'January2026', False, None), (5, P26/'February2026', False, None), (6, P26/'March2026', False, None)]
Q4 = ('2025-10', '2025-11', '2025-12'); Q1 = ('2026-01', '2026-02', '2026-03')
CHUNK = 3_000_000; REF = pd.Timestamp('2026-01-01')
money = lambda s: pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

tgt, feat_frames = [], []
for order, folder, is_feat, _ in SNAPS:
    for ch in pd.read_csv(folder/'calendar.csv', usecols=['listing_id', 'date', 'available'],
                          dtype={'listing_id':'int64','date':'str','available':'str'}, chunksize=CHUNK):
        ym = ch['date'].str.slice(0, 7)
        m1 = ym.isin(Q1)
        if m1.any():
            t = ch[m1].copy(); t['order'] = np.int8(order); t['av'] = (t['available'] == 't')
            tgt.append(t[['listing_id', 'date', 'order', 'av']])
        if is_feat:
            m4 = ym.isin(Q4)
            if m4.any():
                f = ch[m4].copy(); f['order'] = np.int8(order); f['av'] = (f['available'] == 't')
                f['mon'] = f['date'].str.slice(5, 7)
                feat_frames.append(f[['listing_id', 'date', 'mon', 'order', 'av']])
    print('read calendar', folder.name)

# target: Q1 2026 booked (differencing)
tg = pd.concat(tgt, ignore_index=True).sort_values(['listing_id', 'date', 'order'])
g = tg.groupby(['listing_id', 'date'])['av']
ytgt = ((g.first() & (~g.last())).groupby(level='listing_id').sum().astype('int32').clip(0, 92)
        .rename('y_booked_q1_2026'))

# features: Q4 2025 behaviour
cal = pd.concat(feat_frames, ignore_index=True).sort_values(['listing_id', 'date', 'order'])
gd = cal.groupby(['listing_id', 'date'])
dt = pd.DataFrame({'mon': gd['mon'].first(), 'first_av': gd['av'].first(),
                   'last_av': gd['av'].last(), 'changed': gd['av'].first().ne(gd['av'].last())}).reset_index()
dt['booked'] = dt['first_av'] & (~dt['last_av']); dt['occupied'] = ~dt['last_av']
g = dt.groupby('listing_id')
feat = pd.DataFrame({'q3_booked': g['booked'].sum(), 'q3_occupied': g['occupied'].sum(),
                     'q3_obs_days': g['date'].count(), 'q3_changes': g['changed'].sum()})
feat['q3_booking_rate'] = feat['q3_booked'] / feat['q3_obs_days'].clip(lower=1)
feat['q3_occ_rate'] = feat['q3_occupied'] / feat['q3_obs_days'].clip(lower=1)
mb = dt.pivot_table(index='listing_id', columns='mon', values='booked', aggfunc='sum', fill_value=0)
mb.columns = [f'booked_m{c}' for c in mb.columns]
feat = feat.join(mb)
if 'booked_m12' in feat and 'booked_m10' in feat:
    feat['recency_booked_sep'] = feat['booked_m12']
    feat['trend_sep_minus_jul'] = feat['booked_m12'] - feat['booked_m10']
feat = feat.reset_index()

# price dynamics from Oct/Nov/Dec listings
pp = []
for _, folder, is_feat, m in SNAPS:
    if not is_feat: continue
    d = pd.read_csv(folder/'listings.csv', usecols=['id', 'price'], low_memory=False)
    d['price'] = money(d['price']); pp.append(d.rename(columns={'id': 'listing_id', 'price': f'price_{m}'}).set_index('listing_id'))
pr = pd.concat(pp, axis=1); pcols = [c for c in pr.columns if c.startswith('price_')]
pr['lst_price_mean'] = pr[pcols].mean(axis=1); pr['lst_price_std'] = pr[pcols].std(axis=1)
pr['lst_price_last'] = pr.get('price_12', pr[pcols[-1]])
pr['lst_price_cv'] = pr['lst_price_std'] / pr['lst_price_mean'].replace(0, np.nan)
if 'price_12' in pr and 'price_10' in pr:
    pr['lst_price_change'] = (pr['price_12'] - pr['price_10']) / pr['price_10'].replace(0, np.nan)
pr = pr[['lst_price_mean', 'lst_price_std', 'lst_price_last', 'lst_price_cv'] +
        (['lst_price_change'] if 'lst_price_change' in pr else [])].reset_index()
feat = feat.merge(pr, on='listing_id', how='left')

# static: December 2025 listings
keep = ['id','neighbourhood_cleansed','neighbourhood_group_cleansed','latitude','longitude',
        'property_type','room_type','accommodates','bathrooms','bathrooms_text','bedrooms','beds','price',
        'minimum_nights','maximum_nights','host_is_superhost','host_response_rate','host_acceptance_rate',
        'host_listings_count','host_total_listings_count','instant_bookable','number_of_reviews',
        'number_of_reviews_ltm','number_of_reviews_l30d','reviews_per_month','review_scores_rating',
        'review_scores_accuracy','review_scores_cleanliness','review_scores_location','review_scores_value',
        'estimated_occupancy_l365d','estimated_revenue_l365d','license','name','description','host_since','first_review','last_review']
lst = pd.read_csv(P25/'December2025'/'listings.csv', low_memory=False)
keep = [c for c in keep if c in lst.columns]; lst = lst[keep].rename(columns={'id': 'listing_id'})
lst['price'] = money(lst['price'])
for b in ['host_is_superhost', 'instant_bookable']:
    if b in lst: lst[b] = (lst[b] == 't').astype('int8')
for pct in ['host_response_rate', 'host_acceptance_rate']:
    if pct in lst: lst[pct] = pd.to_numeric(lst[pct].astype(str).str.rstrip('%'), errors='coerce')
if 'bathrooms_text' in lst and lst['bathrooms'].isna().all():
    lst['bathrooms'] = pd.to_numeric(lst['bathrooms_text'].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
lst['license_missing'] = lst['license'].isna().astype('int8') if 'license' in lst else 0
lst['has_reviews'] = (lst.get('number_of_reviews', 0) > 0).astype('int8')
for dcol, new in [('host_since','host_age_days'), ('first_review','days_since_first_review'), ('last_review','days_since_last_review')]:
    if dcol in lst: lst[new] = (REF - pd.to_datetime(lst[dcol], errors='coerce')).dt.days
lst['bed_bath_ratio'] = lst['beds'] / lst['bathrooms'].replace(0, np.nan)
lst['price_per_person'] = lst['price'] / lst['accommodates'].replace(0, np.nan)
name_s = lst['name'].fillna('').astype(str) if 'name' in lst else pd.Series('', index=lst.index)
desc_s = lst['description'].fillna('').astype(str) if 'description' in lst else pd.Series('', index=lst.index)
txt = (name_s + ' ' + desc_s).str.lower()
for kw in ['luxury','private','cozy','modern','spacious','view','near','studio']:
    lst[f'kw_{kw}'] = txt.str.contains(kw, regex=False).astype('int8')
lst['title_len'] = name_s.str.len().astype('int32'); lst['title_words'] = name_s.str.split().str.len().fillna(0).astype('int32')
for m in ['review_scores_rating','host_response_rate','reviews_per_month','estimated_occupancy_l365d']:
    if m in lst: lst[f'{m}_missing'] = lst[m].isna().astype('int8')
lst = lst.drop(columns=[c for c in ['bathrooms_text','name','description','host_since','first_review','last_review','license'] if c in lst])
geo = lst[['latitude', 'longitude']].dropna()
km = KMeans(n_clusters=12, random_state=42, n_init=10).fit(geo)
gc = pd.Series(['__nan__']*len(lst), index=lst.index, dtype='object'); gc.loc[geo.index] = ['c'+str(l) for l in km.labels_]
lst['geo_cluster'] = gc

# assemble on January-2026 universe
univ = pd.read_csv(P26/'January2026'/'listings.csv', usecols=['id'])['id'].rename('listing_id').to_frame()
X = (univ.merge(ytgt, on='listing_id', how='left').merge(feat, on='listing_id', how='left').merge(lst, on='listing_id', how='left'))
X['y_booked_q1_2026'] = X['y_booked_q1_2026'].fillna(0).astype('int32')
X.to_parquet(OUT/'features_setupB.parquet', index=False)
print(f'\nfeatures_setupB.parquet rows={len(X):,} cols={X.shape[1]}')
print('target mean', round(X['y_booked_q1_2026'].mean(), 2), '%zero', round((X['y_booked_q1_2026']==0).mean()*100, 1),
      'cold-start', int(X['q3_obs_days'].isna().sum()))
