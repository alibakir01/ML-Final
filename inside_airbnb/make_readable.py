"""
make_readable.py -- produce human-readable summaries/previews of the data WITHOUT
modifying any original file. Everything is written to inside_airbnb/readable/.
For each dataset: a column dictionary (name | dtype | non-null% | example/stats)
and a head preview. Raw Inside Airbnb files are only READ (never written).
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r'C:\Users\Ali\Downloads\AirBnb_Inside\AirBnb_Inside')
OUT = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\outputs')
R = Path(r'C:\Users\Ali\Documents\GitHub\ML-Final\inside_airbnb\readable')
for sub in ['raw', 'features', 'targets', 'results']:
    (R/sub).mkdir(parents=True, exist_ok=True)
ENC = 'utf-8-sig'   # so Excel shows Turkish/UTF-8 correctly


def dictionary(df, path):
    rows = []
    for c in df.columns:
        s = df[c]; nn = s.notna().mean()*100
        if pd.api.types.is_numeric_dtype(s):
            ex = f'min={s.min():.2f} | mean={s.mean():.2f} | max={s.max():.2f}'
        else:
            vals = s.dropna().astype(str).value_counts().head(3).index.tolist()
            ex = ' | '.join(vals)
        rows.append({'column': c, 'dtype': str(s.dtype), 'non_null_%': round(nn, 1), 'example_or_stats': ex})
    pd.DataFrame(rows).to_csv(path, index=False, encoding=ENC)


def readable(df, name, folder, n=25):
    dictionary(df, R/folder/f'{name}__dictionary.csv')
    df.head(n).to_csv(R/folder/f'{name}__head.csv', index=False, encoding=ENC)


# ---------- RAW (read only) ----------
sep = ROOT/'2025_Inside_Airbnb'/'Q3'/'September2025'
lst = pd.read_csv(sep/'listings.csv', low_memory=False)
readable(lst, 'RAW_listings_September2025', 'raw', n=20)
cal = pd.read_csv(sep/'calendar.csv', nrows=200)
readable(cal, 'RAW_calendar_September2025', 'raw', n=25)
rev = pd.read_csv(sep/'reviews.csv', nrows=200)
readable(rev, 'RAW_reviews_September2025', 'raw', n=25)

# ---------- GENERATED feature matrices ----------
for f in ['features_setupA', 'features_setupB', 'features_setupB2Q', 'features_setupC', 'features_setupC_v2']:
    p = OUT/f'{f}.parquet'
    if p.exists():
        readable(pd.read_parquet(p), f, 'features', n=25)

# ---------- GENERATED targets ----------
for f in ['target_q4_2025_clean', 'target_q1_2026_clean', 'target_setupC_clean',
          'target_q4_2025_booking_diff']:
    p = OUT/f'{f}.csv'
    if p.exists():
        readable(pd.read_csv(p), f, 'targets', n=25)

# ---------- copy already-readable result/stat reports ----------
for p in OUT.glob('*_results.txt'):
    shutil.copy(p, R/'results'/p.name)
for p in OUT.glob('*stats*.txt'):
    shutil.copy(p, R/'results'/p.name)
for p in OUT.glob('*summary*.txt'):
    shutil.copy(p, R/'results'/p.name)
for p in OUT.glob('significance_*.txt'):
    shutil.copy(p, R/'results'/p.name)

# ---------- overview ----------
ov = """# VERI REHBERI (okunur ozet) -- inside_airbnb/readable/

Bu klasor SADECE okuma amacli ozetlerdir. Orijinal veriye dokunulmamistir.

## 1) RAW/  -> Ham Inside Airbnb verisi (Eylul 2025 ornek snapshot)
- RAW_listings_September2025__dictionary.csv : listings.csv'nin TUM sutunlari,
  her sutunun dolu yuzdesi, tipi ve ornek/istatistigi. (75+ sutun)
- RAW_listings_September2025__head.csv       : ilk 20 ilan (Excel'de acabilirsin)
- RAW_calendar_September2025__*               : calendar.csv (listing_id,date,available,price...)
  NOT: 'price' sutunu bu veride BOS; 'available' = t/f (musait/degil).
- RAW_reviews_September2025__*                : reviews.csv (listing_id, date)

Toplam veri: 10 aylik snapshot (Tem 2025 -> Nis 2026). Her ay: listings, calendar,
reviews, neighbourhoods. (Bu klasorde temsili olarak Eylul 2025 gosterildi; digerleri
ayni semaya sahip.)

## 2) FEATURES/ -> Bizim urettigimiz ozellik matrisleri
- features_setupA : Q3 2025 gecmis -> Q4 2025 hedefi
- features_setupB : Q3+Q4 2025 -> Q1 2026 (tek ceyrek)
- features_setupB2Q: iki ceyrek gecmis versiyonu
- features_setupC : 7 aylik gecmis (Tem'25-Oca'26) -> Sub-Mar-Nis 2026 (EN IYI)
- features_setupC_v2: + ince recency trajektorisi
Her biri icin __dictionary.csv (sutun aciklamalari) ve __head.csv (ilk 25 satir).

## 3) TARGETS/ -> Hedef degiskenler (tahmin edilen "booked nights")
- target_*_clean.csv : y_clean (temiz hedef) ve y_orig (ilk tanim) yan yana.
- target_q4_2025_booking_diff.csv : ilk differencing hedefi.

## 4) RESULTS/ -> Model sonuclari, istatistikler, anlamlilik testleri (hazir okunur .txt)

## Onemli kavramlar
- Hedef y = bir mulkun ilgili ceyrekte rezerve edilen gece sayisi [0, ~92], zero-inflated.
- 'booked' (differencing): bir tarih snapshot'lar arasi musait->dolu gecisi yaptiysa.
- NIHAI EN IYI MODEL: Setup C, blend R2 = 0.666, MSE = 255.68.
"""
(R/'00_OVERVIEW.md').write_text(ov, encoding='utf-8')
print('readable klasoru olusturuldu:')
for p in sorted(R.rglob('*')):
    if p.is_file():
        print('  ', p.relative_to(R))
