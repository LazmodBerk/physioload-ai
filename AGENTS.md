# PhysioLoad AI Ajan Kuralları

## Bilimsel sınırlar

- Tek RGB görüntüsünden gerçek GRF, CoP, ayaklar arası kuvvet paylaşımı, kas
  kuvveti, eklem içi kuvvet veya klinik risk üretildiğini iddia etmeyin.
- Metrik ölçek doğrulanmadan metre veya N·m çıktısı üretmeyin.
- Dış moment için yük büyüklüğü, uygulama noktası ve kuvvet doğrultusunu zorunlu
  tutun. Sanal yükü gerçek ölçüm gibi etiketlemeyin.
- Yaklaşık CoM çıktısında antropometrik katsayı kaynağı ve izdüşüm düzlemini
  kaydedin.
- Eşikleri doğrulanmış standart gibi sabitlemeyin; validation bölümünde
  kalibre edin ve test bölümünü eşik seçimi için kullanmayın.

## Veri ve deney bütünlüğü

- Kişi veya deneme aynı anda birden fazla train/validation/test bölümünde
  bulunamaz.
- İşlenmiş veri mantığı değişirse yeni sürüm açın; mevcut manifest farklıysa
  üzerine yazmayın.
- Veri seti lisansını ihlal eden ham görüntüleri Git'e veya herkese açık
  çıktılara koymayın.
- Her sonuçta veri sürümü, model sürümü, eşikler, seed ve Git commit bilgisini
  kaydedin.
- Notebook çıktılarındaki büyük ikili verileri commit etmeyin.

## Mimari

- Görüntü işleme, poz kestirimi, geometri, Quality Gate, MARP ve raporlamayı
  ayrı modüllerde tutun.
- Bir ölçümün geçersizliği diğer bağımsız ölçümleri otomatik reddetmemelidir.
- MediaPipe ve RTMPose sonuçlarını ortak isimli landmark şemasına açıkça
  dönüştürmeden doğrudan karşılaştırmayın.
- Windows ve macOS notebook DataLoader işlerinde `default_num_workers()` kullanın.

## Doğrulama

- Değişiklikten sonra en az `python -m pytest -q` ve
  `python scripts/smoke_test.py` çalıştırın.
- Notebook değişikliklerinde JSON parse kontrolü ve en az ilgili notebook için
  `nbconvert --execute` yapın.
- Gerçek veri olmadan sadece sentetik/sözdizimsel doğrulama yapıldıysa bunu
  açıkça belirtin.

