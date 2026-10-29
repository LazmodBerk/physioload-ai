# PhysioLoad AI Sistem Mimarisi

## Araştırma akışı

```text
lisanslı hazır veri
  -> metadata ve kişi/deneme/kamera envanteri
  -> statik kare seçimi
  -> MediaPipe ve RTMPose ortak şeması
  -> geometrik biyomekanik göstergeler
  -> hata yayılımı ve güvenilirlik özellikleri
  -> Quality Gate (çıktı bazında kabul/ret)
  -> MARP yeniden çekim önerisi
  -> referanslı değerlendirme ve açıklanabilir rapor
```

Kod; görüntü işleme, poz kestirimi, biyomekanik hesap, kalite kararı ve
raporlamayı ayrı modüllerde tutar. Her tahmin kaydı en az veri seti, kişi,
deneme, kamera, görüntü, poz modeli, model sürümü, işlenmiş veri sürümü ve Git
commit kimliğini taşımalıdır.

## A0–A4 ablasyonları

- A0: doğrudan poz çıktısı
- A1: kayıt uygunsa zamansal süzme
- A2: anatomik ve geometrik tutarlılık
- A3: birleşik güvenilirlik puanı
- A4: çıktı bazında Quality Gate

Tüm ablasyonlar aynı kişi/deneme ayrımları üzerinde değerlendirilir. Referans
etiketler tahmin girdisi olarak kullanılmaz.

## Çıktı geçerliliği

Eklem açısı gibi izdüşüm temelli değerler uygun görüş düzleminde raporlanabilir.
Metrik moment kolu ve N·m cinsinden dış moment için ölçek, yük büyüklüğü,
uygulama noktası ve kuvvet doğrultusu zorunludur. Bir çıktının reddedilmesi,
koşulları sağlanan diğer çıktıları otomatik olarak geçersiz kılmaz.

