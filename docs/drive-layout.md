# Google Drive Yerleşimi

Ana klasör: [PhysioLoad_AI_Project](https://drive.google.com/drive/folders/1DLr54GatOzAOhFp_3d9VTbXrqCkr3K-4)

```text
PhysioLoad_AI_Project/
├── code/
│   └── physioload-source.zip
├── data/
│   ├── raw/{fit3d,human36m,other}/
│   ├── metadata/
│   ├── interim/static_frames/
│   ├── processed/v1/
│   └── splits/v1/
├── references/
└── users/berk-can/
    ├── models/{checkpoints,calibrators}/
    └── outputs/{predictions,metrics,figures,reports,logs}/
```

`code/physioload-source.zip`, özel GitHub deposu hazır olana kadar Colab için
çalışır kaynak paketidir. Git deposu bağlandığında notebook bootstrap'ı repo
URL'sini tercih eder; veri ve deney çıktıları Drive'da kalmaya devam eder.

