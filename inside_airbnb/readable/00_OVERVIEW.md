# VERI REHBERI (okunur ozet) -- inside_airbnb/readable/

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
