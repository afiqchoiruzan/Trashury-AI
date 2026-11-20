# TRASHURY AI — Waste Classification & Carbon Reduction Model

Repositori khusus untuk **Model Kecerdasan Buatan (Computer Vision & Estimasi Reduksi Emisi $\text{CO}_2\text{e}$)** pada ekosistem **TRASHURY** (*Offline-First Community Waste Bank System*).

Model dilatih menggunakan pendekatan *Transfer Learning* pada **EfficientNet-B0** untuk mengklasifikasikan **12 kategori sampah rumah tangga**, kemudian diekspor ke format **ONNX** agar dapat dijalankan secara *offline* di berbagai platform (Python, Node.js, maupun Web Browser via `onnxruntime-web`).

---

## 1. Struktur Repositori

```text
trashury-ai-model/
├── README.md                              # Dokumentasi model, evaluasi, dan panduan penggunaan
├── requirements.txt                       # Dependensi Python untuk training & inferensi
├── .gitignore                             # Konfigurasi file yang diabaikan Git
├── inference.py                           # Skrip Python siap pakai untuk uji inferensi gambar (.onnx)
├── models/
│   ├── klasifikasi-sampah-v1.onnx         # Graf arsitektur model ONNX (EfficientNet-B0)
│   ├── klasifikasi-sampah-v1.onnx.data    # Bobot parameter model ONNX (15.31 MB)
│   └── labels.txt                         # Daftar 12 kelas urutan indeks output (0..11)
└── notebooks/
    └── Trashury_Training.ipynb            # Notebook eksperimen training, evaluasi & ekspor ONNX
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

## 3. Hasil Perbandingan Arsitektur & Evaluasi

Dilakukan eksperimen perbandingan tiga arsitektur CNN dengan *pretrained weights* ImageNet-1K pada dataset 2.418 citra sampah:

| Arsitektur | Ukuran Model | Validation Accuracy | Waktu Inferensi | Status |
|---|:---:|:---:|:---:|---|
| **ResNet-18** | $42.73\text{ MB}$ | $84.75\%$ | $\sim 0.45\text{ ms}$ | Baseline |
| **ResNet-34** | $81.35\text{ MB}$ | $86.25\%$ | $\sim 0.72\text{ ms}$ | Baseline |
| **EfficientNet-B0** | **$15.64\text{ MB}$** | **$91.00\%$** | **$\sim 0.37\text{ ms}$** | **Model Terpilih** 🏆 |

### Rincian Evaluasi per Kelas (EfficientNet-B0 — Test Set)

| Kategori Sampah | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Cardboard Boxes** | 0.88 | 0.95 | 0.91 | 22 |
| **Glass Bottles** | 0.74 | 0.88 | 0.80 | 16 |
| **Glass Shards** | 0.92 | 0.92 | 0.92 | 12 |
| **Jug** | 1.00 | 0.93 | 0.97 | 15 |
| **Metal Can** | 0.83 | 0.96 | 0.89 | 26 |
| **Paper** | 1.00 | 0.75 | 0.86 | 12 |
| **Paper Bag** | 0.89 | 0.80 | 0.84 | 20 |
| **Paper Cup** | 0.92 | 0.92 | 0.92 | 12 |
| **Plastic Bottle** | 0.95 | 0.87 | 0.91 | 23 |
| **Plastic Cup** | 0.91 | 0.95 | 0.93 | 21 |
| **Plastic bag** | 0.90 | 0.95 | 0.92 | 20 |
| **food waste** | 1.00 | 0.83 | 0.91 | 18 |
| **Rata-rata Makro / Akurasi** | **0.91** | **0.90** | **0.90** | **217** |

---

## 4. Spesifikasi Teknis Model ONNX

* **Berkas Model:** `models/klasifikasi-sampah-v1.onnx` (+ `klasifikasi-sampah-v1.onnx.data`)
* **Input Tensor:**
  * Nama: `"input"`
  * Shape: `[batch_size, 3, 224, 224]` (`float32`, format channel `RGB` / `NCHW`)
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
