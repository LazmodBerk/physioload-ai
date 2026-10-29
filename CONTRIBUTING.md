# Katkı Kılavuzu

Ana geliştirme dalı `develop` olarak kabul edilir. Yeni çalışma için
`feature/<kısa-ad>` dalı açın; küçük, izlenebilir commit'ler kullanın.

## Değişiklik kontrol listesi

1. Bilimsel kapsam ve `AGENTS.md` sınırları korunuyor.
2. Yeni bağımlılık `requirements.txt` ve gerekirse `pyproject.toml` içinde.
3. Veri ön işleme değiştiyse yeni işlenmiş veri sürümü tanımlandı.
4. Kişi/deneme veri sızıntısı testi geçiyor.
5. `python -m pytest -q` ve smoke test geçiyor.
6. Notebook'lar temiz, sıralı ve sabit seed kullanıyor.

Checkpoint, lisanslı veri ve sır dosyaları Git'e eklenmez. Uzun deney çıktıları
Drive'daki `users/<kişi>/` ağacına yazılır.

