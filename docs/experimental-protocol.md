# Deney Protokolü

## Ana araştırma sorusu

Tek kamera landmark hatası eklem açısı, moment kolu ve koşullu dış moment
hesabına nasıl yayılır; sistem hangi koşullarda sayısal çıktıyı reddetmelidir?

## Birincil değerlendirme

- Landmark: PCK ve konum hatası
- Eklem açısı: derece cinsinden MAE ve RMSE
- Ağır hata ayrımı: AUROC ve AUPRC
- Seçici tahmin: kabul oranı, kabul edilen örneklerde hata ve AURC
- MARP: yeniden çekim sonrası hata azalması, başarılı düzeltme oranı ve tekrar sayısı

## Veri ayrımı

Kişi bazında train, validation ve test ayrımı kullanılır. Aynı kişinin farklı
kamera veya denemeleri farklı bölümlere geçemez. Eşikler yalnız validation
bölümünde belirlenir; test bölümü tek seferlik nihai değerlendirmeye ayrılır.

## Baseline karşılaştırmaları

1. Standart kamera talimatı
2. Yalnız poz güveni düşükse genel yeniden çekim
3. Birleşik Quality Gate
4. Mekanik hata nedenine göre MARP eylemi

Sayısal eşikler önceden doğrulanmış standart gibi sunulmaz. Literatür,
validasyon dağılımı ve danışman kararıyla gerekçelendirilir.

