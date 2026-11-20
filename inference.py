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


def preprocess_image_edge(image_path: str) -> np.ndarray:
    """Membaca gambar, mengubah ukuran ke 260x260, dan menerapkan normalisasi ImageNet untuk EfficientNet-B2."""
    img = Image.open(image_path).convert("RGB").resize((260, 260))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(logits - np.max(logits))
    return exp_scores / np.sum(exp_scores)


def extract_siglip_embedding(
    image_path: str,
    backbone_id: str = "google/siglip-so400m-patch14-384",
) -> np.ndarray:
    """Mengekstrak embedding 1152-dim menggunakan Frozen SigLIP-SO400M Vision Model."""
    import torch
    from torchvision import transforms
    from transformers import SiglipImageProcessor, SiglipVisionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = SiglipImageProcessor.from_pretrained(backbone_id)
    vision_model = SiglipVisionModel.from_pretrained(backbone_id).to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])
    img_tensor = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = vision_model(pixel_values=img_tensor).pooler_output.float().cpu().numpy()
    return emb


def _format_hasil(probs: np.ndarray, berat_kg: float, top_k: int) -> dict[str, Any]:
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


def prediksi_sampah(
    image_path: str,
    mode: str = "edge",
    model_path: str | None = None,
    berat_kg: float = 1.0,
    top_k: int = 3,
) -> dict[str, Any]:
    """Menjalankan prediksi sampah dalam Mode 'edge' (EfficientNet-B2) atau 'siglip' (SigLIP-SO400M + MLP2 Ensemble)."""
    if mode == "edge":
        target_model = model_path or "models/klasifikasi-sampah-v3.onnx"
        if not os.path.exists(target_model):
            raise FileNotFoundError(f"File model Edge tidak ditemukan: {target_model}")

        session = ort.InferenceSession(target_model)
        input_name = session.get_inputs()[0].name
        tensor = preprocess_image_edge(image_path)
        logits = session.run(None, {input_name: tensor})[0][0]
        probs = softmax(logits)
        return _format_hasil(probs, berat_kg, top_k)

    if mode == "siglip":
        target_model = model_path or "models/siglip_ens_mlp2_head.onnx"
        if not os.path.exists(target_model):
            raise FileNotFoundError(f"File head ONNX SigLIP tidak ditemukan: {target_model}")

        emb = extract_siglip_embedding(image_path)
        session = ort.InferenceSession(target_model)
        input_name = session.get_inputs()[0].name
        # Output siglip_ens_mlp2_head.onnx sudah berupa probabilitas rata-rata Softmax 5 seed
        probs = session.run(None, {input_name: emb.astype(np.float32)})[0][0]
        return _format_hasil(probs, berat_kg, top_k)

    raise ValueError(f"Mode tidak dikenal: '{mode}'. Gunakan 'edge' atau 'siglip'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferensi Klasifikasi Sampah TRASHURY (Edge & High-Accuracy)")
    parser.add_argument("--image", type=str, required=True, help="Path ke file gambar sampah")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["edge", "siglip"],
        default="edge",
        help="Mode inferensi: 'edge' (EfficientNet-B2 29.4MB) atau 'siglip' (SigLIP-SO400M + MLP2 Ensemble)",
    )
    parser.add_argument("--model", type=str, default=None, help="Path kustom ke file model ONNX")
    parser.add_argument("--berat", type=float, default=1.0, help="Berat timbangan sampah dalam kg")
    args = parser.parse_args()

    hasil = prediksi_sampah(args.image, mode=args.mode, model_path=args.model, berat_kg=args.berat)

    print(f"=== HASIL PREDIKSI AI TRASHURY [Mode: {args.mode.upper()}] ===")
    for i, k in enumerate(hasil["kandidat_teratas"], start=1):
        tanda = " [REKOMENDASI UTAMA]" if i == 1 else ""
        print(
            f"  #{i} {k['nama']} ({k['label']}) — Confidence: {k['confidence']*100:.2f}%{tanda}\n"
            f"     Estimasi Nilai ({args.berat} kg): Rp {k['estimasi_rupiah']:,} | "
            f"Reduksi Emisi: {k['estimasi_co2e_kg']:.4f} kg CO2e"
        )

    for p in hasil["peringatan_kontaminasi"]:
        print(f"\n⚠️ {p}")
