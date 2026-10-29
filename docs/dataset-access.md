# Veri Seti Erişimi ve Yerleşimi

Fit3D ve Human3.6M lisanslı veri setleridir. Bu depo veri dosyalarını indirmez,
yeniden dağıtmaz veya Drive'a otomatik kopyalamaz. Erişim başvurusu, lisans ve
kullanım koşulları araştırmacı tarafından resmi kaynaklardan tamamlanmalıdır.

## Beklenen yerleşim

```text
data/raw/fit3d/
data/raw/human36m/
data/raw/other/
data/metadata/dataset_inventory.csv
```

`dataset_inventory.csv` için önerilen sütunlar:

```text
image_id,dataset,subject_id,trial_id,camera_id,image_path,pose_label,reference_path,view_label,frame_index
```

İşlenmiş veriler yeni ön işleme mantığında yeni bir sürüme yazılmalıdır.
Mevcut `manifest.json` ile parametreler uyuşmazsa üzerine yazma yapılmaz.

## Gizlilik

- Lisansın izin vermediği görüntüler depoya, rapora veya herkese açık bağlantıya
  eklenmez.
- Kişi kimlikleri gerekiyorsa takma kimlikle kaydedilir.
- Yalnızca türetilmiş metrikler ve paylaşılması izinli örnekler raporlanır.

