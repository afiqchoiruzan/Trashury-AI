# TRASHURY AI — Waste Classification & Carbon Reduction Model

Repositori khusus untuk **Model Kecerdasan Buatan (Computer Vision & Estimasi Reduksi Emisi $\text{CO}_2\text{e}$)** pada ekosistem **TRASHURY** (*Offline-First Community Waste Bank System*).

Model dilatih menggunakan pendekatan *Two-Stage Fine-Tuning* pada **EfficientNet-B2** (resolusi `260×260`) untuk mengklasifikasikan **12 kategori sampah rumah tangga**, kemudian diekspor menjadi satu berkas utuh berformat **ONNX** agar dapat dijalankan secara *offline* di berbagai platform (Python, Node.js, maupun Web Browser via `onnxruntime-web`).

---

## 1. Struktur Repositori

```text
trashury-ai-model/
├── README.md                              # Dokumentasi model, evaluasi, dan panduan penggunaan
├── requirements.txt                       # Dependensi Python untuk training & inferensi
├── .gitignore                             # Konfigurasi file yang diabaikan Git
├── inference.py                           # Skrip Python siap pakai untuk uji inferensi gambar (.onnx)
├── models/
│   ├── klasifikasi-sampah-v3.onnx         # Model ONNX tunggal (EfficientNet-B2, 29.41 MB)
│   └── labels.txt                         # Daftar 12 kelas urutan indeks output (0..11)
└── notebooks/
    └── Trashury_v3.ipynb                  # Notebook training Two-Stage Fine-Tuning, TTA, & ekspor ONNX
```

---

## 2. Daftar 12 Kelas Sampah, Harga Acuan, & Faktor Emisi $\text{CO}_2\text{e}$

| Indeks | Label Model (`labels.txt`) | Nama Kategori Lokal | Harga Acuan / kg | Faktor Reduksi $\text{CO}_2\text{e}$ / kg |
|:---:|---|---|:---:|:---:|
| **0** | `Cardboard Boxes` | Kardus / Karton | Rp 1.500 | $1.10\text{ kg CO}_2\text{e}$ |
| **1** | `Glass Bottles` | Botol Kaca Utuh | Rp 500 | $0.35\text{ kg CO}_2\text{e}$ |
| **2** | `Glass Shards` | Pecahan Kaca *(Hazard)* | Rp 200 | $0.30\text{ kg CO}_2\text{e}$ |
| **3** | `Jug` | Jeriken / Botol HDPE | Rp 3.000 | $1.80\text{ kg CO}_2\text{e}$ |
| **4** | `Metal Can` | Kaleng / Logam | Rp 4.000 | $4.00\text{ kg CO}_2\text{e}$ |
| **5** | `Paper` | Kertas / HVS / Buku | Rp 1.200 | $1.30\text{ kg CO}_2\text{e}$ |
| **6** | `Paper Bag` | Kantong Kertas | Rp 1.000 | $1.00\text{ kg CO}_2\text{e}$ |
| **7** | `Paper Cup` | Gelas Kertas | Rp 500 | $0.50\text{ kg CO}_2\text{e}$ |
| **8** | `Plastic Bottle` | Botol Plastik (PET) | Rp 3.500 | $2.10\text{ kg CO}_2\text{e}$ |
| **9** | `Plastic Cup` | Gelas Plastik (PP) | Rp 2.500 | $1.70\text{ kg CO}_2\text{e}$ |
| **10** | `Plastic bag` | Kantong Plastik / Kresek | Rp 500 | $1.20\text{ kg CO}_2\text{e}$ |
| **11** | `food waste` | Sampah Makanan / Organik *(Kontaminasi)* | Rp 0 | $0.25\text{ kg CO}_2\text{e}$ |

---

## 3. Hasil Perbandingan Arsitektur & Evaluasi (`v3` Two-Stage Fine-Tuning)

Dilakukan eksperimen perbandingan tiga arsitektur keluarga EfficientNet pada resolusi `260×260` menggunakan *Two-Stage Fine-Tuning*, `AdamW`, `CosineAnnealingLR`, `Label Smoothing (0.1)`, serta evaluasi *Test-Time Augmentation (TTA)* pada dataset 2.418 citra sampah (`1556 train / 400 valid / 207 test`):

| Arsitektur | Valid Acc | Test Std | **Test TTA** | Ukuran (`.pth`) | Latency (GPU) | Status |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **EfficientNet-B0** | $92.75\%$ | $86.47\%$ | $86.96\%$ | $15.64\text{ MB}$ | $1.23\text{ ms}$ | Kandidat |
| **EfficientNet-B2** | $92.75\%$ | **$88.41\%$** | **$88.89\%$** | $29.88\text{ MB}$ | **$1.00\text{ ms}$** | **Model Terpilih** 🏆 |
| **EfficientNetV2-S** | $95.00\%$ | $87.44\%$ | $87.92\%$ | $77.90\text{ MB}$ | $1.11\text{ ms}$ | Kandidat |

**Alasan Pemilihan:** **EfficientNet-B2** memberikan akurasi generalisasi terbaik pada data uji nyata (**88.89%** Test Accuracy + TTA) dengan ukuran berkas ONNX tunggal yang ramping (**29.41 MB**).

### Rincian Evaluasi per Kelas (`EfficientNet-B2` — Test Set + TTA)

| Kategori Sampah | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Cardboard Boxes** | 0.91 | 0.91 | 0.91 | 22 |
| **Glass Bottles** | 0.76 | 0.81 | 0.79 | 16 |
| **Glass Shards** | 1.00 | 1.00 | 1.00 | 12 |
| **Jug** | 0.93 | 0.93 | 0.93 | 15 |
| **Metal Can** | 0.93 | 0.96 | 0.94 | 26 |
| **Paper** | 0.75 | 0.75 | 0.75 | 12 |
| **Paper Bag** | 0.91 | 0.95 | 0.93 | 22 |
| **Paper Cup** | 0.79 | 0.88 | 0.83 | 17 |
| **Plastic Bottle** | 0.87 | 0.83 | 0.85 | 24 |
| **Plastic Cup** | 0.95 | 0.82 | 0.88 | 22 |
| **Plastic bag** | 0.89 | 0.89 | 0.89 | 9 |
| **food waste** | 1.00 | 0.90 | 0.95 | 10 |
| **Rata-rata Makro / Akurasi** | **0.89** | **0.89** | **0.89** | **207** |

---

## 4. Spesifikasi Teknis Model ONNX

* **Berkas Model:** `models/klasifikasi-sampah-v3.onnx` (`29.41 MB`, 1 berkas utuh dengan `export_params=True` & `dynamo=False`)
* **Input Tensor:**
  * Nama: `"input"`
  * Shape: `[batch_size, 3, 260, 260]` (`float32`, format channel `RGB` / `NCHW`)
  * Normalisasi: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`
* **Output Tensor:**
  * Nama: `"output"`
  * Shape: `[batch_size, 12]` (`float32`, *raw logits* — aplikasikan fungsi **Softmax** untuk mendapatkan probabilitas $0..1$).

---

## 5. Cara Menjalankan Inferensi (Python)

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan prediksi pada gambar sampah (beserta simulasi berat kg)
python inference.py --image contoh_sampah.jpg --berat 2.5
```

---

## Penulis
**Muhammad Afiq Mirza Choiruzan** — *AI Engineer, Tim TRASHURY (DTETI FT UGM)*
