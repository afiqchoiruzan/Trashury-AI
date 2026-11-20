import argparse
import os
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

# Daftar 12 kategori sampah beserta metadata harga dan faktor reduksi emisi CO2e (IPCC/KLHK)
KATEGORI_SAMPAH: list[dict[str, Any]] = [
    {"index": 0, "label": "Cardboard Boxes", "nama": "Kardus / Karton", "harga_per_kg": 1500, "co2e_per_kg": 1.10},
    {"index": 1, "label": "Glass Bottles", "nama": "Botol Kaca Utuh", "harga_per_kg": 500, "co2e_per_kg": 0.35},
    {"index": 2, "label": "Glass Shards", "nama": "Pecahan Kaca", "harga_per_kg": 200, "co2e_per_kg": 0.30},
    {"index": 3, "label": "Jug", "nama": "Jeriken / Botol HDPE", "harga_per_kg": 3000, "co2e_per_kg": 1.80},
    {"index": 4, "label": "Metal Can", "nama": "Kaleng / Logam", "harga_per_kg": 4000, "co2e_per_kg": 4.00},
    {"index": 5, "label": "Paper", "nama": "Kertas / HVS / Buku", "harga_per_kg": 1200, "co2e_per_kg": 1.30},
    {"index": 6, "label": "Paper Bag", "nama": "Kantong Kertas", "harga_per_kg": 1000, "co2e_per_kg": 1.00},
    {"index": 7, "label": "Paper Cup", "nama": "Gelas Kertas", "harga_per_kg": 500, "co2e_per_kg": 0.50},
    {"index": 8, "label": "Plastic Bottle", "nama": "Botol Plastik (PET)", "harga_per_kg": 3500, "co2e_per_kg": 2.10},
    {"index": 9, "label": "Plastic Cup", "nama": "Gelas Plastik (PP)", "harga_per_kg": 2500, "co2e_per_kg": 1.70},
    {"index": 10, "label": "Plastic bag", "nama": "Kantong Plastik / Kresek", "harga_per_kg": 500, "co2e_per_kg": 1.20},
    {"index": 11, "label": "food waste", "nama": "Sampah Makanan / Organik", "harga_per_kg": 0, "co2e_per_kg": 0.25},
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str) -> np.ndarray:
    """Membaca gambar, mengubah ukuran ke 224x224, dan menerapkan normalisasi ImageNet."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    # Ubah dari HWC (224, 224, 3) ke NCHW (1, 3, 224, 224)
    tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return tensor


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(logits - np.max(logits))
    return exp_scores / np.sum(exp_scores)


def prediksi_sampah(
    image_path: str,
    model_path: str = "models/klasifikasi-sampah-v1.onnx",
    berat_kg: float = 1.0,
    top_k: int = 3,
) -> dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    tensor = preprocess_image(image_path)

    logits = session.run(None, {input_name: tensor})[0][0]
    probs = softmax(logits)

    urut_idx = np.argsort(probs)[::-1][:top_k]
    kandidat_teratas = []
    for idx in urut_idx:
        kat = KATEGORI_SAMPAH[int(idx)]
        conf = float(probs[idx])
        kandidat_teratas.append({
            **kat,
            "confidence": conf,
            "estimasi_rupiah": int(round(berat_kg * kat["harga_per_kg"])),
            "estimasi_co2e_kg": round(berat_kg * kat["co2e_per_kg"], 4),
        })

    peringatan = []
    if any(k["index"] == 11 and k["confidence"] >= 0.2 for k in kandidat_teratas):
        peringatan.append("PERINGATAN: Terdeteksi sampah organik / sisa makanan (food waste).")
    if any(k["index"] == 2 and k["confidence"] >= 0.2 for k in kandidat_teratas):
        peringatan.append("PERINGATAN: Terdeteksi pecahan kaca (Glass Shards). Gunakan sarung tangan.")

    return {
        "rekomendasi_utama": kandidat_teratas[0],
        "kandidat_teratas": kandidat_teratas,
        "peringatan_kontaminasi": peringatan,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferensi Klasifikasi Sampah TRASHURY (ONNX)")
    parser.add_argument("--image", type=str, required=True, help="Path ke file gambar sampah")
    parser.add_argument("--model", type=str, default="models/klasifikasi-sampah-v1.onnx", help="Path ke model ONNX")
    parser.add_argument("--berat", type=float, default=1.0, help="Berat timbangan sampah dalam kg")
    args = parser.parse_args()

    hasil = prediksi_sampah(args.image, args.model, args.berat)
    utama = hasil["rekomendasi_utama"]

    print("=== HASIL PREDIKSI AI TRASHURY (Top-3 Smart Suggestion) ===")
    for i, k in enumerate(hasil["kandidat_teratas"], start=1):
        tanda = " [REKOMENDASI UTAMA]" if i == 1 else ""
        print(
            f"  #{i} {k['nama']} ({k['label']}) — Confidence: {k['confidence']*100:.2f}%{tanda}\n"
            f"     Estimasi Nilai ({args.berat} kg): Rp {k['estimasi_rupiah']:,} | "
            f"Reduksi Emisi: {k['estimasi_co2e_kg']:.4f} kg CO2e"
        )

    for p in hasil["peringatan_kontaminasi"]:
        print(f"\n⚠️ {p}")
