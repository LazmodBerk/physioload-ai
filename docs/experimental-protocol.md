# Deney Protokolü

## Ana araştırma sorusu

Tek RGB görüntüsünde landmark konum hatalarının farklı statik biyomekanik
ölçümlere etkisi, ölçüm bazlı seçici güvenilirlik kararlarını iyileştirebilir mi?

## Deney sırası

1. Fit3D ve Human3.6M erişim, lisans, kamera ve referans uygunluğunu doğrula.
2. MediaPipe ile RTMPose modellerini aynı görüntüler ve ortak landmark şemasıyla
   karşılaştır.
3. Referans 3B noktaları ilgili kamera düzlemine izdüşürerek eşdeğer 2B
   referans ölçümler oluştur.
4. Seçilen landmarklara gerekçelendirilmiş büyüklük ve yönlerde kontrollü
   bozunum uygula.
5. Her ölçüm için Ölçüm Duyarlılık Profili ve Hata Etki Haritası üret.
6. Fiziksel geçerlilik ön denetimini uygula.
7. A0–A4 aşamalarını aynı veri ayrımlarında değerlendir.

## Birincil ölçütler

- Landmark konumu: karşılaştırılabilir koordinatlarda konum hatası ve PCK
- Eklem açısı: derece cinsinden MAE ve RMSE
- Ağır hata ayrımı: AUROC ve AUPRC
- Seçici tahmin: kabul oranı, kabul edilen ölçümlerin hatası ve AURC
- Açıklanabilirlik: doğru durum ve gerekçe kodu oranı
- Hata Etki Haritası: en etkili landmark sıralamasının bozunum deneyleriyle
  tutarlılığı

## Veri ayrımı ve eşik seçimi

Kişi ve deneme bazında train, validation ve test ayrımı kullanılır. Aynı kişinin
farklı kamera veya denemeleri farklı bölümlere geçemez. Kalite Kapısı eşikleri
yalnız validation bölümünde seçilir. Test bölümü nihai değerlendirmeye ayrılır.

## A0–A4 karşılaştırmaları

1. A0 doğrudan poz ve ölçüm
2. A1 uygun ardışık kayıtta zamansal süzme
3. A2 geometrik tutarlılık
4. A3 birleşik güvenilirlik puanı
5. A4 fiziksel geçerlilik ve ölçüm duyarlılığı kullanan Kalite Kapısı

Yalnız model güvenini kullanan temel yöntem ayrıca raporlanır. Gerçek deney
sonuçları oluşmadan yeni mimarinin daha iyi olduğu varsayılmaz. Sayısal eşikler
literatür, veri uygunluğu, validasyon dağılımı ve danışman kararıyla
gerekçelendirilir.

