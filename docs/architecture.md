# PhysioLoad AI — Ölçüm Duyarlı Seçici Biyomekanik Analiz Mimarisi

Bu belge, proje planındaki G1–G10 gereksinimlerini, 14 haftalık mühendislik tasarımı kapsamını ve projenin bilimsel özgünlük hedefini gerçekleştiren mimari tasarımı tanımlar.

---

## Önerilen Temel Yaklaşım

> **"Önce ölç, sonra ölçümün güvenilirliğini sınayarak raporla."**

Her biyomekanik çıktı kendi fiziksel geçerlilik koşulları, hata duyarlılığı ve güvenilirlik düzeyine göre ayrı değerlendirilir. Güvenilmeyen sonuçlar, diğer geçerli sonuçlar korunarak reddedilir.

- **Ölçüme duyarlı:** Her açı ve gösterge için ayrı hata analizi
- **Fiziksel olarak tutarlı:** Hesaplanamayacak fiziksel değerleri üretmeme
- **Açıklanabilir:** Kabul ve ret kararlarının nedenlerini gösterme

---

## 1. Neden Bu Mimari?

| Mimari Yaklaşımı | Özgünlük Potansiyeli | Uygulanabilirlik | Değerlendirme |
| :--- | :--- | :--- | :--- |
| Hazır poz modeli + açı hesaplama | Düşük | Çok yüksek | Tek başına yetersiz |
| Hazır poz modeli + genel güven puanı | Orta | Yüksek | İyi, ancak sınırlı |
| **Hazır poz modeli + ölçüme özgü hata yayılımı + fiziksel geçerlilik + seçici kabul/ret** | **Yüksek** | **Yüksek** | **Önerilen mimari** |
| Sıfırdan 3B biyomekanik yapay zekâ modeli eğitimi | Belirsiz | Düşük | Mevcut kapsam için gereksiz |

### Asıl Özgünlük Nerede? (Örnek: Dirsek Açısı)
- **Klasik yaklaşımda:** Model noktaları buldu $\rightarrow$ Dirsek açısı hesaplandı $\rightarrow$ Sonuç gösterildi.
- **PhysioLoad AI mimarisinde:** Model noktaları buldu $\rightarrow$ Dirsek açısı hesaplandı $\rightarrow$ Noktalardaki konum hatasının açıyı ne kadar etkileyebileceği incelendi $\rightarrow$ Görünürlük ve geometrik tutarlılık denetlendi $\rightarrow$ Kamera bakış açısının ölçüme uygunluğu değerlendirildi $\rightarrow$ Sonuç kabul veya reddedildi $\rightarrow$ Kararın gerekçesi raporlandı.

Dirsek açısı reddedilirken aynı görüntüdeki geçerli bir gövde veya bacak ölçümü kabul edilir.

---

## 2. Bütünleşik Yedi Katmanlı Mimari ve Ayrı Doğrulama Hattı

```text
KATMAN 1: Veri ve Görüntü Yönetimi
  Hazır RGB görüntüler · kişi/kamera/deneme kayıtları · uygunluk denetimi
     ↓
KATMAN 2: Poz Kestirimi ve Ortak Veri Biçimi
  MediaPipe Pose veya RTMPose · ortak eklem koordinatları · model güveni
     ↓
KATMAN 3: Biyomekanik Hesaplama Motoru
  2B açılar · segment yönelimleri · asimetri · yaklaşık CoM · koşullu dış moment
     ↓
KATMAN 4: Hata Yayılımı ve Ölçüm Duyarlılığı
  Poz hatası ölçümü ne kadar değiştiriyor? (Ölçüm Duyarlılık Profili & U_j)
     ↓
KATMAN 5: Fiziksel Geçerlilik Denetimi
  Bu ölçüm bu girdilerden hesaplanabilir mi? (Zorunlu fiziksel kısıtlar)
     ↓
KATMAN 6: Ölçüm Bazlı Kalite Kapısı (ÖZGÜN ÇEKİRDEK)
  Görünürlük + güven + bakış açısı + geometri + hata duyarlılığı
  [KABUL / RET / HESAPLANAMAZ]
     ↓
KATMAN 7: Açıklanabilir Biyomekanik Raporlama
  Ölçüm · birim · geçerlilik · karar gerekçesi · hata göstergesi

AYRI DOĞRULAMA HATTI:
  Referans veriler → Hata analizi → A0–A4 Karşılaştırması (referanslar modele tahmin girdisi olamaz)
```

---

## 3. Katmanların Teknik Tasarımı

### Katman 1 — Veri Yönetimi ve Statik Görüntü Hazırlama
- **Veri Kümeleri:** Fit3D ve Human3.6M temel adaylardır. OpenCap yöntemsel dayanak olarak incelenir.
- **Standart Kayıt Yapısı:** `image_id`, `dataset`, `subject_id`, `trial_id`, `camera_id`, `frame_index`, `image_path`, `pose_label`, `view_label`, `reference_path`, `reference_role` (`evaluation_only`), `static_eligibility`.
- **Kritik Kural:** Referans eklem verisi tahmin sırasında poz modeline verilmez; yalnızca değerlendirme hattında kullanılır.

### Katman 2 — Yapay Zekâ Poz Kestirimi
- **Modeller:** MediaPipe Pose (Heavy) ve RTMPose/MMPose.
- **Mimari Sadeleştirme:** İki model sistemde her zaman birlikte çalışmaz. Deney aşamasında aynı protokolle karşılaştırılır; ana model seçilir. Sistem model değiştirmeyi destekleyen ortak adaptör arayüzü (`physioload_17`) korur.
- **Model Güveni:** MediaPipe görünürlük ile RTMPose güveni aynı anlama gelmez; model bazında kalibre edilir.

### Katman 3 — Statik Biyomekanik Hesaplama Motoru
Yalnızca matematiksel fonksiyonları çalıştırır; güvenilirlik kararı vermez:
- **2B Eklem Açıları:** $\theta = \arccos\left(\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)$ (Diz, kalça, dirsek, omuz)
- **Segment Yönelimleri:** $\phi = \operatorname{arctan2}(\Delta y, \Delta x)$
- **Bilateral Asimetri:** $|\theta_{\text{sol}} - \theta_{\text{sağ}}|$
- **Yaklaşık 2B CoM İzdüşümü:** Winter (2009) antropometrik segment kütle katsayıları
- **Dış Moment Katkısı:** $M = \mathbf{r} \times \mathbf{F}$ (yalnızca doğrulanmış ölçek ve yük ile)

### Katman 4 — Ölçüme Özgü Hata Yayılımı ve Duyarlılık Profili
Poz kestiriminden elde edilen eklem noktaları $P$, biyomekanik fonksiyon $f_j$:
$$\hat{y}_j = f_j(P)$$
Kontrollü konum bozunumu $\delta$:
$$\hat{y}_j' = f_j(P + \delta) \quad \implies \quad \Delta y_j = |\hat{y}_j' - \hat{y}_j|$$
Ölçüm belirsizliği göstergesi (Monte Carlo quantile özeti):
$$U_j = \operatorname{Quantile}_q\left(\Delta y_j^{(1)}, \ldots, \Delta y_j^{(K)}\right)$$
- **Hata Etki Haritası:** Hangi eklem noktasının bozunumunun ilgili ölçümde en büyük değişime yol açtığını normalize ederek gösterir.

### Katman 5 — Fiziksel Geçerlilik Denetimi
Modelin güveni yüksek olsa bile fiziksel girdiler eksikse değer üretilmez:
1. İstenen ölçüm ve gerekli girdiler tanımlı mı?
2. Görüntü geometrisi ve projeksiyon düzlemi uygun mu?
3. Metrik ölçek ve yük tanımlı mı? (Dış moment için zorunlu)
- Uygunsa $\rightarrow$ Duyarlılık ve güvenilirlik puanına geç.
- Uygun değilse $\rightarrow$ `NOT_COMPUTABLE` olarak işaretle, değer üretme, nedenini kaydet.

### Katman 6 — Ölçüm Bazlı Kalite Kapısı (Özgün Çekirdek)
Kalite Kapısı her ölçüm için iki aşamada ayrı karar verir:
$$Q_j = g(C_j, V_j, B_j, G_j, U_j)$$
- $C_j$: Noktaların model tahmin güveni
- $V_j$: Noktaların görünürlük bilgisi
- $B_j$: Kamera bakış açısı ve ölçüm düzlemi uygunluğu
- $G_j$: Geometrik ve anatomik tutarlılık
- $U_j$: İlgili ölçümün hata duyarlılığı

#### Üç Durumlu Karar Sözleşmesi:
1. **KABUL (`ACCEPTED`):** Fiziksel geçerlilik ve güvenilirlik koşulları karşılanmıştır. Sonuç kendi birimi, düzlemi ve sınırlılıklarıyla sunulur.
2. **RET (`REJECTED`):** Ölçüm teknik olarak hesaplanabilir; ancak nokta kestirimi veya hata duyarlılığı nedeniyle güvenilmezdir. Sayısal değer raporda sunulmaz; ret gerekçesi gösterilir.
3. **HESAPLANAMAZ (`NOT_COMPUTABLE`):** Gerekli fiziksel büyüklük, ölçek, koordinat veya geometri yoktur. Güven puanı yüksek olsa dahi fiziksel sonuç üretilmez.

### Katman 7 — Açıklanabilir Biyomekanik Raporlama
Her göstergeyi değer, birim, projeksiyon düzlemi, durum (`ACCEPTED`/`REJECTED`/`NOT_COMPUTABLE`), karar gerekçesi ve hata göstergesiyle şeffaf şekilde sunar.

---

## 4. Bilimsel Doğrulama ve A0–A4 Karşılaştırması

| Aşama | İçerik | Doğrulama Amacı |
| :--- | :--- | :--- |
| **A0** | Doğrudan poz kestirimi ve açı hesabı | Temel hata düzeyini belirlemek |
| **A1** | Uygun ardışık kayıt varsa zamansal süzme | Süzmenin hata üzerindeki etkisini incelemek |
| **A2** | Geometrik tutarlılık denetimi | Anatomik geometrinin katkısını sınamak |
| **A3** | Birleşik güvenilirlik puanı | Model güveninden daha yararlı olup olmadığını görmek |
| **A4** | **Ölçüm Bazlı Kalite Kapısı** | Hatalı ölçümleri seçici reddetmenin ve bağımsız ölçümleri korumanın faydasını sınamak |

- **Temel Değerlendirme:** Hata–Kabul Oranı ilişkisi (AURC), AUROC, AUPRC, Açı MAE/RMSE.
- **Kritik İlke:** Referans 3B açı ile 2B açı doğrudan karşılaştırılmaz; 3B referans noktalar kamera düzlemine izdüşürülerek 2B eşdeğer referans oluşturulur.

---

## 5. Akademik Özgünlük İfadesi

> "Bu projede, tek RGB görüntülerinden elde edilen hazır poz kestirim sonuçlarının statik biyomekanik göstergelere dönüştürülmesi sırasında oluşabilecek hataların ölçüm bazında değerlendirilmesini sağlayan modüler bir sistem mimarisi tasarlanmaktadır. Önerilen mimarinin temel katkısı; eklem noktası kestirim güveni, görünürlük, kamera bakış açısı, geometrik tutarlılık ve kontrollü hata yayılımı bulgularını her biyomekanik çıktının fiziksel geçerlilik koşullarıyla birlikte değerlendiren açıklanabilir bir Kalite Kapısı mekanizmasıdır. Bu mekanizma, güvenilirliği yetersiz ölçümlerin gerekçeli olarak reddedilmesini ve diğer geçerli ölçümlerin korunmasını amaçlamaktadır. Yaklaşımın katkısı, yalnızca poz kestirim güvenine dayalı temel yöntemlerle karşılaştırmalı deney protokolü üzerinden değerlendirilecektir."
