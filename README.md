# PhysioLoad AI

PhysioLoad AI, tek RGB görüntüsünden statik duruşun seçilmiş biyomekanik
göstergelerini hesaplayan ve her çıktının fiziksel hesaplanabilirliğini, hata
duyarlılığını ve güvenilirliğini ayrı değerlendiren bir araştırma projesidir.
Sistem gerçek yer tepki kuvveti, basınç merkezi, kas
kuvveti veya klinik tanı ürettiğini iddia etmez. Ölçek, görüş açısı ya da dış
yük tanımı doğrulanamıyorsa ilgili mekanik çıktı Quality Gate tarafından
reddedilir.

## Çalışma modeli

FaceQR ile aynı çalışma ilkeleri korunur:

- Kod yerel Git çalışma ağacında veya Colab runtime'ında çalışır.
- Veri, checkpoint ve deney çıktıları Google Drive'da kalıcıdır.
- İşlenmiş veri `data/processed/<sürüm>` ve ayrımlar
  `data/splits/<sürüm>` altında manifest ile sürümlenir.
- Colab çıktıları `users/berk-can/` altında kişiye özel tutulur.
- Notebook'lar yerel Jupyter, VS Code yerel kernel, VS Code Colab ve tarayıcı
  Colab ortamlarında aynı akışla çalışır.

Google Drive klasörü: [PhysioLoad_AI_Project](https://drive.google.com/drive/folders/1DLr54GatOzAOhFp_3d9VTbXrqCkr3K-4)

## Klasör yapısı

```text
physioload_env.py              ortam, Drive, kullanıcı ve sürüm yönetimi
physioload/                    yeniden kullanılabilir araştırma kodu
configs/                       deney ve Quality Gate ayarları
data/
  raw/{fit3d,human36m,other}   lisanslı ham veri; Git'e girmez
  metadata/                    kişi/deneme/kamera/referans envanteri
  interim/static_frames/       aday sabit kareler
  processed/<version>/         ortak biçime dönüştürülmüş veri
  splits/<version>/            kişi/deneme ayrımları
models/checkpoints/            yerel checkpoint'ler
outputs/{predictions,metrics,figures,reports,logs}
notebooks/                     deney akışı
scripts/                       kurulum, doğrulama ve Drive paketi
tests/                         geometrik ve kalite kapısı testleri
docs/                          mimari, veri erişimi ve kullanım kılavuzları
```

## Notebook sırası

1. `00_environment_check.ipynb` — ortam, GPU, Drive ve yollar
2. `01_dataset_inventory.ipynb` — Fit3D/Human3.6M envanteri ve lisans kaydı
3. `02_static_frame_selection.ipynb` — düşük hareketli sabit kare seçimi
4. `03_pose_model_benchmark.ipynb` — MediaPipe ve RTMPose karşılaştırması
5. `04_biomechanics.ipynb` — açı, yönelim, asimetri, yaklaşık CoM ve koşullu moment
6. `05_uncertainty_propagation.ipynb` — landmark bozunumu ve Monte Carlo analizi
7. `06_quality_gate_training.ipynb` — ölçüm duyarlılığı destekli güvenilirlik kalibrasyonu
8. `07_physical_validity_and_impact_map.ipynb` — fiziksel geçerlilik ve Hata Etki Haritası
9. `08_evaluation_reporting.ipynb` — A0–A4, üç durumlu kararlar ve risk-kapsama raporu

Ana karar sırası fiziksel geçerlilik, ölçüm duyarlılığı ve Ölçüm Bazlı Kalite
Kapısıdır. Çıktılar `ACCEPTED`, `REJECTED` veya `NOT_COMPUTABLE` durumlarından
biriyle ve gerekçe kodlarıyla raporlanır.

## Hızlı kurulum

Windows PowerShell:

```powershell
.\scripts\init.ps1
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Linux/macOS/Colab benzeri Bash ortamı:

```bash
./scripts/init.sh
source .venv/bin/activate
jupyter lab
```

Sadece iskeleti hızlı doğrulamak için:

```powershell
python scripts/smoke_test.py
python -m pytest -v  # 49 test passed
```

VS Code + Colab adımları için [kullanım kılavuzuna](docs/vscode-colab-kullanim.md)
bakın.

## Bilimsel sınırlar

- Tek görüntüden doğrulanmış GRF, ayaklar arası yük paylaşımı veya gerçek CoP
  üretilmez.
- Yaklaşık CoM antropometrik varsayımlara dayalı konumsal bir kestirimdir.
- N·m hesabı yalnızca metrik ölçek, yük kütlesi, uygulama noktası ve kuvvet
  doğrultusu tanımlandığında yapılır.
- Sanal yük deneyi analitik doğrulamadır; gerçek ölçüm olarak raporlanmaz.
- Sayısal eşikler literatür, veri uygunluğu ve validasyon sonuçlarıyla
  belirlenecek; yapılandırmadaki başlangıç değerleri bilimsel standart olarak
  sunulmayacaktır.

