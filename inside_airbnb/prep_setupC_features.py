"""
prep_setupC_features.py -- Setup C features: 7-month history (Jul 2025 - Jan 2026)
to predict Feb-Mar-Apr 2026. Assembled from existing matrices (no heavy reads):
  H_Q3 (Jul-Sep) from features_setupA, H_Q4 (Oct-Dec) from features_setupB,
  H_Jan (Jan 2026) from cached q1_panel, + QoQ/7-month trends, + Jan-2026 static
  + Jan-2026 review velocity. Universe = January 2026 listings.
Output: features_setupC_feats.parquet (target merged later).
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
P26 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2026_Inside_Airbnb')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
money = lambda s: pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

# --- H_Q3 (Jul-Sep 2025) from Setup A ---
a = pd.read_parquet(OUT/'features_setupA.parquet')
hq3 = a[['listing_id','q3_booked','q3_occupied','q3_obs_days','q3_changes','q3_booking_rate','q3_occ_rate']].copy()
hq3.columns = ['listing_id'] + ['hq3_'+c for c in ['booked','occupied','obs_days','changes','booking_rate','occ_rate']]
# --- H_Q4 (Oct-Dec 2025) from Setup B ---
b = pd.read_parquet(OUT/'features_setupB.parquet')
hq4 = b[['listing_id','q3_booked','q3_occupied','q3_obs_days','q3_changes','q3_booking_rate','q3_occ_rate']].copy()
hq4.columns = ['listing_id'] + ['hq4_'+c for c in ['booked','occupied','obs_days','changes','booking_rate','occ_rate']]
# --- H_Jan (Jan 2026) from cached q1 panel ---
pan = pd.read_parquet(OUT/'q1_panel.parquet')
jan = pan[pan['date'].str.slice(0, 7) == '2026-01']
g = jan.sort_values(['listing_id','date','order']).groupby(['listing_id','date'])['av']
jd = pd.DataFrame({'first': g.first(), 'last': g.last()}).reset_index()
jd['booked'] = jd['first'] & (~jd['last']); jd['occ'] = ~jd['last']
gj = jd.groupby('listing_id')
hjan = pd.DataFrame({'hjan_booked': gj['booked'].sum(), 'hjan_occ': gj['occ'].sum(),
                     'hjan_obs': gj['date'].count()}).reset_index()
hjan['hjan_booking_rate'] = hjan['hjan_booked'] / hjan['hjan_obs'].clip(lower=1)

# --- assemble + trends ---
univ = pd.read_csv(P26/'January2026'/'listings.csv', usecols=['id'])['id'].rename('listing_id').to_frame()
X = univ.merge(hq3, on='listing_id', how='left').merge(hq4, on='listing_id', how='left').merge(hjan, on='listing_id', how='left')
X['h7_booked_total'] = X[['hq3_booked','hq4_booked','hjan_booked']].sum(axis=1)
X['trend_q4_q3'] = X['hq4_booking_rate'] - X['hq3_booking_rate']
X['trend_jan_q4'] = X['hjan_booking_rate'] - X['hq4_booking_rate']
X['momentum'] = X['hjan_booking_rate'] - X['hq3_booking_rate']
X['recency_booking_rate'] = X['hjan_booking_rate']

# --- static + geo + text from Jan 2026 listings ---
keep = ['id','neighbourhood_cleansed','neighbourhood_group_cleansed','latitude','longitude','property_type',
        'room_type','accommodates','bathrooms','bathrooms_text','bedrooms','beds','minimum_nights','maximum_nights',
        'host_is_superhost','host_response_rate','host_acceptance_rate','host_listings_count','host_total_listings_count',
        'instant_bookable','number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d','reviews_per_month',
        'review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_location',
        'review_scores_value','estimated_occupancy_l365d','estimated_revenue_l365d','license','name','description',
        'host_since','first_review','last_review']
lst = pd.read_csv(P26/'January2026'/'listings.csv', low_memory=False)
keep = [c for c in keep if c in lst.columns]; lst = lst[keep].rename(columns={'id': 'listing_id'})
for bcol in ['host_is_superhost','instant_bookable']:
    if bcol in lst: lst[bcol] = (lst[bcol] == 't').astype('int8')
for pct in ['host_response_rate','host_acceptance_rate']:
    if pct in lst: lst[pct] = pd.to_numeric(lst[pct].astype(str).str.rstrip('%'), errors='coerce')
if 'bathrooms_text' in lst and lst['bathrooms'].isna().all():
    lst['bathrooms'] = pd.to_numeric(lst['bathrooms_text'].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
lst['license_missing'] = lst['license'].isna().astype('int8') if 'license' in lst else 0
lst['has_reviews'] = (lst.get('number_of_reviews', 0) > 0).astype('int8')
REF = pd.Timestamp('2026-02-01')
for dcol, new in [('host_since','host_age_days'),('first_review','days_since_first_review'),('last_review','days_since_last_review')]:
    if dcol in lst: lst[new] = (REF - pd.to_datetime(lst[dcol], errors='coerce')).dt.days
lst['bed_bath_ratio'] = lst['beds'] / lst['bathrooms'].replace(0, np.nan)
name_s = lst['name'].fillna('').astype(str) if 'name' in lst else pd.Series('', index=lst.index)
desc_s = lst['description'].fillna('').astype(str) if 'description' in lst else pd.Series('', index=lst.index)
txt = (name_s + ' ' + desc_s).str.lower()
for kw in ['luxury','private','cozy','modern','spacious','view','near','studio']:
    lst[f'kw_{kw}'] = txt.str.contains(kw, regex=False).astype('int8')
lst['title_len'] = name_s.str.len().astype('int32'); lst['title_words'] = name_s.str.split().str.len().fillna(0).astype('int32')
for m in ['review_scores_rating','host_response_rate','reviews_per_month','estimated_occupancy_l365d']:
    if m in lst: lst[f'{m}_missing'] = lst[m].isna().astype('int8')
lst = lst.drop(columns=[c for c in ['bathrooms_text','name','description','host_since','first_review','last_review','license'] if c in lst])
geo = lst[['latitude','longitude']].dropna()
km = KMeans(n_clusters=12, random_state=42, n_init=10).fit(geo)
gc = pd.Series(['__nan__']*len(lst), index=lst.index, dtype='object'); gc.loc[geo.index] = ['c'+str(l) for l in km.labels_]
lst['geo_cluster'] = gc
X = X.merge(lst, on='listing_id', how='left')

# --- review velocity (Jan 2026 reviews, cutoff Feb 1 2026) ---
r = pd.read_csv(P26/'January2026'/'reviews.csv', usecols=['listing_id','date'])
r['date'] = pd.to_datetime(r['date'], errors='coerce'); r = r[r['date'] < REF]
age = (REF - r['date']).dt.days; r = r.assign(age=age); g = r.groupby('listing_id')
rev = pd.DataFrame({'rev_total': g.size(),
                    'rev_90': g['age'].apply(lambda x: (x < 90).sum()),
                    'rev_365': g['age'].apply(lambda x: (x < 365).sum()),
                    'rev_recency_days': g['age'].min()}).reset_index()
X = X.merge(rev, on='listing_id', how='left')
X.to_parquet(OUT/'features_setupC_feats.parquet', index=False)
print('features_setupC_feats.parquet rows=%d cols=%d' % (X.shape[0], X.shape[1]))
print('has H_Q3:%d H_Q4:%d H_Jan:%d' % (X['hq3_booked'].notna().sum(), X['hq4_booked'].notna().sum(), X['hjan_booked'].notna().sum()))
