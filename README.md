# TRASHURY AI — Waste Classification & Carbon Reduction Models

Repositori khusus untuk **Model Kecerdasan Buatan (Computer Vision & Estimasi Reduksi Emisi $\text{CO}_2\text{e}$)** pada ekosistem **TRASHURY** (*Offline-First Community Waste Bank System*).

Repositori ini menyediakan **dua mode arsitektur AI** untuk mengklasifikasikan **12 kategori sampah rumah tangga** dalam format **ONNX**:

1. **Mode 1 — High-Accuracy Server/Cloud (`SigLIP-SO400M + ResMLP / MLP2 Ensemble`)** 🏆  
   * **Test Accuracy:** **`96.03%`** | **Test Macro F1:** **`0.9644`** (pada `353` citra uji, seluruh 12 kelas memiliki F1 $\ge 0.90$).
   * Memanfaatkan *Frozen Vision Encoder* `google/siglip-so400m-patch14-384` (`1152`-dim) dengan *Feature-Space Mixup*, `RMSNorm`, dan *Multi-Seed Ensemble*.
2. **Mode 2 — Lightweight Edge Offline PWA (`EfficientNet-B2`)** ⚡  
   * **Test Accuracy (TTA):** **`88.89%`** | **Valid Accuracy:** **`92.75%`** | **Ukuran:** **`29.41 MB`** (1 berkas `.onnx` utuh).
   * Dirancang ringan agar berjalan 100% *offline* di peramban kasir PWA via `onnxruntime-web`.

---

## 1. Struktur Repositori

```text
trashury-ai-model/
├── README.md                              # Dokumentasi arsitektur, evaluasi, dan panduan deploy
├── app.py                                 # FastAPI Cloud AI Server (SigLIP-SO400M + MLP2 ONNX)
├── Dockerfile                             # Docker image siap deploy ke Hugging Face Spaces (Port 7860)
├── requirements-api.txt                   # Dependensi ringan khusus Cloud API (CPU-only)
├── requirements.txt                       # Dependensi lengkap Python untuk training & inferensi CLI
├── .gitignore                             # Konfigurasi file yang diabaikan Git
├── inference.py                           # Skrip Python CLI untuk inferensi (--mode edge | --mode siglip)
├── models/
│   ├── klasifikasi-sampah-v3.onnx         # Mode 2 (Edge): EfficientNet-B2 ONNX tunggal (29.41 MB)
│   ├── siglip_ens_mlp2_head.onnx          # Mode 1 (SigLIP): Head ONNX Multi-Seed MLP2 + Norm (12.59 MB)
│   ├── siglip_heads_checkpoints.zip       # Mode 1 (SigLIP): 5 checkpoint PyTorch (.pt) seed 42..46
│   └── labels.txt                         # Daftar 12 kelas urutan indeks output (0..11)
└── notebooks/
    ├── Trashury_SigLIP_Probe.ipynb        # Notebook training & evaluasi SigLIP-SO400M + ResMLP/MLP2 (96.03%)
    └── Trashury_v3.ipynb                  # Notebook training EfficientNet-B2 Two-Stage Fine-Tuning (88.89%)
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

## 3. Ringkasan Evaluasi & Perbandingan Model

### Mode 1: High-Accuracy `SigLIP-SO400M + ResMLP / MLP2` (`3.164` Citra: Roboflow + TrashBox `70/15/15`)
* **Dataset:** `2.259` train (multi-view embedding), `552` valid, `353` test (bebas duplikat `dHash` 8×8).
* **Valid Accuracy / Macro F1:** **`96.56%`** / **`0.9661`** (`ResMLP`) | **`96.20%`** / **`0.9642`** (`Multi-Seed MLP2`).
* **Test Accuracy / Macro F1:** **`96.03%`** / **`0.9644`** (naik **`+7.14%`** dibanding `EfficientNet-B2`).

| Kategori Sampah | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Cardboard Boxes** | 1.00 | 1.00 | **1.00** | 32 |
| **Glass Bottles** | 0.97 | 0.92 | **0.94** | 36 |
| **Glass Shards** | 1.00 | 1.00 | **1.00** | 12 |
| **Jug** | 0.97 | 1.00 | **0.99** | 35 |
| **Metal Can** | 1.00 | 0.97 | **0.99** | 40 |
| **Paper** | 0.97 | 1.00 | **0.98** | 32 |
| **Paper Bag** | 1.00 | 0.95 | **0.98** | 22 |
| **Paper Cup** | 0.90 | 1.00 | **0.95** | 37 |
| **Plastic Bottle** | 0.88 | 0.93 | **0.90** | 30 |
| **Plastic Cup** | 0.97 | 0.89 | **0.93** | 38 |
| **Plastic bag** | 0.93 | 0.90 | **0.91** | 29 |
| **food waste** | 1.00 | 1.00 | **1.00** | 10 |
| **Rata-rata Makro / Akurasi** | **0.97** | **0.96** | **0.96** | **353** |

---

### Mode 2: Lightweight Edge `EfficientNet-B2` (`2.163` Citra Roboflow, `260×260` Two-Stage Fine-Tuning)

| Arsitektur | Valid Acc | Test Std | **Test TTA** | Ukuran (`.pth`) | Latency (GPU) | Status |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **EfficientNet-B0** | $92.75\%$ | $86.47\%$ | $86.96\%$ | $15.64\text{ MB}$ | $1.23\text{ ms}$ | Kandidat |
| **EfficientNet-B2** | $92.75\%$ | **$88.41\%$** | **$88.89\%$** | $29.88\text{ MB}$ | **$1.00\text{ ms}$** | **Model Edge Terpilih** ⚡ |
| **EfficientNetV2-S** | $95.00\%$ | $87.44\%$ | $87.92\%$ | $77.90\text{ MB}$ | $1.11\text{ ms}$ | Kandidat |

---

## 4. Spesifikasi Teknis ONNX

| Mode | File ONNX | Input Tensor | Output Tensor |
|---|---|---|---|
| **Edge (`EfficientNet-B2`)** | `models/klasifikasi-sampah-v3.onnx` (`29.41 MB`) | `input`: `[batch, 3, 260, 260]` (`RGB`, normalisasi ImageNet) | `output`: `[batch, 12]` (*raw logits*, gunakan **Softmax**) |
| **SigLIP Head (`MLP2 Ensemble`)** | `models/siglip_ens_mlp2_head.onnx` (`12.59 MB`) | `embedding`: `[batch, 1152]` (output `SiglipVisionModel`) | `probabilities`: `[batch, 12]` (sudah Softmax rata-rata 5 seed) |

---

## 5. Cara Menjalankan Inferensi (Python)

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Mode Edge Offline (EfficientNet-B2 — ringan, langsung dari gambar)
python inference.py --mode edge --image contoh_sampah.jpg --berat 2.5

# 3. Mode High-Accuracy (SigLIP-SO400M + Multi-Seed MLP2 Head)
python inference.py --mode siglip --image contoh_sampah.jpg --berat 2.5
```

---

## 6. Deployment Cloud AI ke Hugging Face Spaces (Hybrid Cloud-Edge)

**Live Demo & Cloud API:** [huggingface.co/spaces/afiqmc777/trashury-siglip-api](https://huggingface.co/spaces/afiqmc777/trashury-siglip-api) (`https://afiqmc777-trashury-siglip-api.hf.space`)

Arsitektur **Hybrid Cloud-Edge** TRASHURY menjalankan:
* **Cloud AI (Online):** `app.py` (`SigLIP-SO400M + MLP2 Ensemble`, akurasi **`96.03%`**) di-host gratis pada **Hugging Face Spaces** (*Gradio SDK + ZeroGPU Free*), sehingga **tidak membebani RAM 4 GiB VM Azure DTETI**.
* **Edge AI (Offline / Fallback):** `models/klasifikasi-sampah-v3.onnx` (`EfficientNet-B2`, `29.41 MB`, akurasi **`88.89%`**) berjalan langsung di browser PWA via `onnxruntime-web` apabila perangkat sedang offline atau Cloud API timeout (> 5 detik).

### Langkah Deploy ke Hugging Face Spaces (100% Gratis — Gradio SDK)
1. Buka [huggingface.co/new-space](https://huggingface.co/new-space) dan buat Space baru:
   * **Space name:** `trashury-siglip-api`
   * **SDK:** pilih **Gradio** → template **Blank** *(Gratis — jangan pilih Docker karena berbayar)*
   * **Hardware:** pilih **ZeroGPU (Free)** *(Gratis untuk akun Free tier; kode `app.py` sudah memakai `@spaces.GPU`)*
2. Upload atau `git push` file berikut ke repositori Space tersebut:
   * `app.py`
   * `requirements-space.txt` *(ubah nama menjadi `requirements.txt` di Space)*
   * `models/siglip_ens_mlp2_head.onnx`
   * `models/labels.txt`
3. Space aktif di `https://afiqmc777-trashury-siglip-api.hf.space`:
   * **Web Demo UI:** [huggingface.co/spaces/afiqmc777/trashury-siglip-api](https://huggingface.co/spaces/afiqmc777/trashury-siglip-api)
   * **Gradio API Endpoint:** `/predict` (via `@gradio/client` atau `gradio_client`)

### Contoh Integrasi Frontend JavaScript/TypeScript (`@gradio/client`)
```ts
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("afiqmc777/trashury-siglip-api");
const result = await client.predict("/predict", {
  image: handle_file(imageBlob),
  berat_kg: 2.5,
});
// result.data[1] berisi JSON detail prediksi (mode, kandidat_teratas, peringatan_kontaminasi, estimasi_rupiah, estimasi_co2e_kg)
console.log(result.data[1]);
```

---

## Penulis
**Muhammad Afiq Mirza Choiruzan** — *AI Engineer, Tim TRASHURY (DTETI FT UGM)*
