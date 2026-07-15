"""
readable_2026.py -- readable previews of the ORIGINAL Jan/Feb/Mar/Apr 2026 data.
Reads only (never modifies). Writes previews to inside_airbnb/readable/raw_2026/<Month>/.
Per month: listings head+dictionary, calendar sample+summary, reviews head.
"""
from pathlib import Path
import numpy as np, pandas as pd
P26 = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside\2026_Inside_Airbnb')
R = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\readable\raw_2026')
MONTHS = ['January2026', 'February2026', 'March2026', 'April2026']
ENC = 'utf-8-sig'
money = lambda s: pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

def diction(df, path):
    rows = []
    for c in df.columns:
        s = df[c]; nn = s.notna().mean()*100
        if pd.api.types.is_numeric_dtype(s):
            ex = f'min={s.min():.2f} | mean={s.mean():.2f} | max={s.max():.2f}'
        else:
            ex = ' | '.join(s.dropna().astype(str).value_counts().head(3).index.tolist())
        rows.append({'column': c, 'dtype': str(s.dtype), 'non_null_%': round(nn, 1), 'example_or_stats': ex})
    pd.DataFrame(rows).to_csv(path, index=False, encoding=ENC)

for m in MONTHS:
    d = R/m; d.mkdir(parents=True, exist_ok=True)
    # listings
    lst = pd.read_csv(P26/m/'listings.csv', low_memory=False)
    lst.head(30).to_csv(d/'listings_head.csv', index=False, encoding=ENC)
    diction(lst, d/'listings_dictionary.csv')
    # calendar: head + summary (huge file -> sample)
    cal_head = pd.read_csv(P26/m/'calendar.csv', nrows=50)
    cal_head.to_csv(d/'calendar_sample.csv', index=False, encoding=ENC)
    cols = list(cal_head.columns)
    use = [c for c in ['listing_id', 'date', 'available', 'price'] if c in cols]
    samp = pd.read_csv(P26/m/'calendar.csv', usecols=use, dtype=str, nrows=1_000_000)
    first_date = samp['date'].iloc[0]
    tail = pd.read_csv(P26/m/'calendar.csv', usecols=['date']).iloc[-1]['date']
    price_line = (f'price DOLU orani (ilk 1M): {money(samp["price"]).notna().mean()*100:.1f}%'
                  if 'price' in use else 'price sutunu YOK (bu snapshot fiyatsiz)')
    summ = [f'{m} calendar.csv ozet',
            f'sutunlar: {cols}',
            f'tarih araligi (yaklasik): {first_date}  ->  {tail}',
            f'available dagilimi (ilk 1M satir): ' + str(samp['available'].value_counts().to_dict()),
            price_line,
            f'benzersiz listing (ilk 1M): {samp["listing_id"].nunique():,}']
    (d/'calendar_summary.txt').write_text('\n'.join(summ), encoding='utf-8')
    # reviews
    rev = pd.read_csv(P26/m/'reviews.csv')
    rev.head(30).to_csv(d/'reviews_head.csv', index=False, encoding=ENC)
    (d/'reviews_summary.txt').write_text(
        f'{m} reviews.csv: satir={len(rev):,} | tarih {rev["date"].min()} -> {rev["date"].max()}',
        encoding='utf-8')
    print(f'{m}: listings {lst.shape}, reviews {len(rev):,} -> yazildi')
print('\nHepsi: inside_airbnb/readable/raw_2026/<Ay>/')
