# VS Code ile Colab Üzerinde PhysioLoad Eğitimi

Google Colab, Python ve Jupyter uzantıları bu bilgisayarda kurulu durumdadır.
VS Code'da `google.colab` uzantısı kullanılacaktır.

## Yerleşim

```text
Yerel VS Code                  Colab runtime                 Google Drive
notebooks/*.ipynb  -------->  /content/physioload-ai  <---- PhysioLoad_AI_Project
proje kodu                     GPU / geçici disk             data ve users/berk-can
```

## İlk bağlantı

1. Bu klasörü VS Code ile açın.
2. `notebooks/00_environment_check.ipynb` dosyasını açın.
3. Sağ üstte kernel seçiciyi açıp Colab çalışma zamanını seçin.
4. Google hesabıyla oturum açın ve gerekiyorsa GPU runtime seçin.
5. İlk hücreyi çalıştırın; Drive bağlama isteğini onaylayın.

İlk hücre önce `PHYSIOLOAD_REPO_URL` değişkenini kontrol eder. Özel GitHub deposu
hazır değilse Drive'daki `code/physioload-source.zip` anlık görüntüsünü açar. Bu
sayede sistem şimdi çalışır. GitHub deposu açıldığında `.env` veya Colab Secrets
üzerinden repo URL'si tanımlanarak FaceQR'daki kod-Git/veri-Drive düzenine tam
geçiş yapılır.

## Günlük kullanım

Notebook'ları numara sırasıyla çalıştırın. Veri ve ayrımlar ortak klasörde,
checkpoint ve çıktılar `users/berk-can/` altında saklanır. Runtime kopsa bile
Drive çıktıları kalır.

Eğitimden önce şunları doğrulayın:

- `context` satırı `colab (vscode)` veya `colab`
- `device` satırı GPU isteniyorsa `cuda`
- `artifacts root` PhysioLoad Drive klasörü
- `processed ver` doğru veri sürümü

## GitHub'a geçiş

Yerel proje bir Git çalışma ağacı olarak hazırlanmıştır. Özel depo URL'si
oluşturulduğunda:

```powershell
git remote add origin <özel-depo-url>
git push -u origin develop
```

Sonra `PHYSIOLOAD_REPO_URL` ve gerekirse `PHYSIOLOAD_REPO_BRANCH=develop`
tanımlanır. Özel depoda Colab için FaceQR ile aynı read-only deploy-key yöntemi
önerilir; gizli anahtar notebook'a veya Git'e eklenmez.

