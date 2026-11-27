import io
import os
import time
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

try:
    import spaces

    zerogpu = spaces.GPU(duration=15)
except ImportError:
    def zerogpu(fn):
        return fn

BACKBONE_ID = os.getenv("SIGLIP_BACKBONE_ID", "google/siglip-so400m-patch14-384")
HEAD_ONNX_PATH = os.getenv("SIGLIP_HEAD_ONNX", "models/siglip_ens_mlp2_head.onnx")

# 12 Kategori sampah TRASHURY (100% sinkron dengan @trashury/shared/kategoriSampah.ts)
MASTER_KATEGORI_SAMPAH: list[dict[str, Any]] = [
    {
        "id_kategori": "00000000-0000-4000-8000-000000000001",
        "nama_kategori": "Kardus / Karton",
        "label_model": "Cardboard Boxes",
        "harga_per_kg": 1500,
        "faktor_co2e_per_kg": 1.10,
        "aktif": True,
        "deskripsi": "Kardus bergelombang, box kemasan cokelat kering",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000002",
        "nama_kategori": "Botol Kaca Utuh",
        "label_model": "Glass Bottles",
        "harga_per_kg": 500,
        "faktor_co2e_per_kg": 0.35,
        "aktif": True,
        "deskripsi": "Botol sirup, kecap, atau minuman kaca dalam kondisi utuh",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000003",
        "nama_kategori": "Pecahan Kaca",
        "label_model": "Glass Shards",
        "harga_per_kg": 200,
        "faktor_co2e_per_kg": 0.30,
        "aktif": True,
        "deskripsi": "Belahan atau pecahan kaca (butuh penanganan hati-hati)",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000004",
        "nama_kategori": "Jeriken / Botol HDPE",
        "label_model": "Jug",
        "harga_per_kg": 3000,
        "faktor_co2e_per_kg": 1.80,
        "aktif": True,
        "deskripsi": "Wadah plastik tebal, jeriken minyak, botol shampo",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000005",
        "nama_kategori": "Kaleng / Logam",
        "label_model": "Metal Can",
        "harga_per_kg": 4000,
        "faktor_co2e_per_kg": 4.00,
        "aktif": True,
        "deskripsi": "Kaleng minuman aluminium, kaleng biskuit atau seng",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000006",
        "nama_kategori": "Kertas / HVS / Buku",
        "label_model": "Paper",
        "harga_per_kg": 1200,
        "faktor_co2e_per_kg": 1.30,
        "aktif": True,
        "deskripsi": "Kertas dokumen HVS, majalah, koran, dan buku tulis",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000007",
        "nama_kategori": "Kantong Kertas",
        "label_model": "Paper Bag",
        "harga_per_kg": 1000,
        "faktor_co2e_per_kg": 1.00,
        "aktif": True,
        "deskripsi": "Paper bag belanja dan kantong kertas kraft",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000008",
        "nama_kategori": "Gelas Kertas",
        "label_model": "Paper Cup",
        "harga_per_kg": 500,
        "faktor_co2e_per_kg": 0.50,
        "aktif": True,
        "deskripsi": "Gelas kopi kertas sekali pakai",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000009",
        "nama_kategori": "Botol Plastik (PET)",
        "label_model": "Plastic Bottle",
        "harga_per_kg": 3500,
        "faktor_co2e_per_kg": 2.10,
        "aktif": True,
        "deskripsi": "Botol air mineral transparan bening atau biru muda",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000010",
        "nama_kategori": "Gelas Plastik (PP)",
        "label_model": "Plastic Cup",
        "harga_per_kg": 2500,
        "faktor_co2e_per_kg": 1.70,
        "aktif": True,
        "deskripsi": "Gelas plastik air mineral / minuman boba",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000011",
        "nama_kategori": "Kantong Plastik / Kresek",
        "label_model": "Plastic bag",
        "harga_per_kg": 500,
        "faktor_co2e_per_kg": 1.20,
        "aktif": True,
        "deskripsi": "Kantong kresek belanjaan LDPE/HDPE tipis",
    },
    {
        "id_kategori": "00000000-0000-4000-8000-000000000012",
        "nama_kategori": "Sampah Makanan / Organik",
        "label_model": "food waste",
        "harga_per_kg": 0,
        "faktor_co2e_per_kg": 0.25,
        "aktif": True,
        "deskripsi": "Sisa makanan organik (bahan kompos / pakan ternak)",
    },
]

state: dict[str, Any] = {
    "vision_model": None,
    "image_mean": None,
    "image_std": None,
    "onnx_session": None,
    "onnx_input_name": None,
    "device": "cpu",
}


def _load_models() -> None:
    if state.get("vision_model") is not None and state.get("onnx_session") is not None:
        return

    # Pada Hugging Face ZeroGPU, hindari inisialisasi CUDA di main process agar tidak error & hemat kuota GPU.
    use_cuda = os.getenv("USE_CUDA", "0") == "1" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    proc = SiglipImageProcessor.from_pretrained(BACKBONE_ID)
    vision_model = SiglipVisionModel.from_pretrained(BACKBONE_ID).to(device).eval()

    if not os.path.exists(HEAD_ONNX_PATH):
        raise FileNotFoundError(f"File head ONNX tidak ditemukan: {HEAD_ONNX_PATH}")

    onnx_session = ort.InferenceSession(HEAD_ONNX_PATH, providers=["CPUExecutionProvider"])

    state["vision_model"] = vision_model
    state["image_mean"] = np.array(proc.image_mean, dtype=np.float32)
    state["image_std"] = np.array(proc.image_std, dtype=np.float32)
    state["onnx_session"] = onnx_session
    state["onnx_input_name"] = onnx_session.get_inputs()[0].name
    state["device"] = str(device)


def _preprocess_siglip(image: Image.Image) -> torch.Tensor:
    img_rgb = image.convert("RGB").resize((384, 384), Image.Resampling.BICUBIC)
    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    arr = (arr - state["image_mean"]) / state["image_std"]
    chw = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return torch.from_numpy(chw).to(state["device"])


def run_prediction(image: Image.Image, berat_kg: float = 1.0, top_k: int = 3) -> dict[str, Any]:
    _load_models()
    t0 = time.perf_counter()
    pixel_values = _preprocess_siglip(image)
    with torch.inference_mode():
        embedding = state["vision_model"](pixel_values=pixel_values).pooler_output.float().cpu().numpy()

    probs = state["onnx_session"].run(
        None,
        {state["onnx_input_name"]: embedding.astype(np.float32)},
    )[0][0]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    urut_idx = np.argsort(probs)[::-1][:top_k]
    kandidat_teratas: list[dict[str, Any]] = []
    for idx in urut_idx:
        i = int(idx)
        kat = MASTER_KATEGORI_SAMPAH[i]
        conf = float(probs[i])
        kandidat_teratas.append(
            {
                "index": i,
                "label_model": kat["label_model"],
                "kategori": kat,
                "confidence": conf,
                "estimasi_rupiah": int(round(berat_kg * kat["harga_per_kg"])),
                "estimasi_co2e_kg": round(berat_kg * kat["faktor_co2e_per_kg"], 4),
            }
        )

    peringatan_kontaminasi: list[str] = []
    if any(k["index"] == 11 and k["confidence"] >= 0.2 for k in kandidat_teratas):
        peringatan_kontaminasi.append(
            "Terdeteksi sampah organik / sisa makanan (food waste). Mohon pisahkan dari tumpukan daur ulang."
        )
    if any(k["index"] == 2 and k["confidence"] >= 0.2 for k in kandidat_teratas):
        peringatan_kontaminasi.append(
            "Terdeteksi pecahan kaca (Glass Shards). Harap gunakan sarung tangan pelindung saat menimbang."
        )

    terbaik = kandidat_teratas[0]
    return {
        "mode": "cloud-siglip",
        "latency_ms": latency_ms,
        **terbaik,
        "kandidat_teratas": kandidat_teratas,
        "peringatan_kontaminasi": peringatan_kontaminasi,
    }


def health_endpoint() -> dict[str, Any]:
    ready = state.get("vision_model") is not None and state.get("onnx_session") is not None
    return {
        "status": "ok" if ready else "loading",
        "mode": "cloud-siglip",
        "model_loaded": ready,
        "device": state.get("device", "cpu"),
        "backbone": BACKBONE_ID,
    }


async def predict_endpoint(
    file: UploadFile = File(..., description="Foto sampah (JPEG/PNG/WebP)"),
    berat_kg: float = Form(1.0, ge=0.0, description="Berat timbangan sampah dalam kilogram"),
    top_k: int = Form(3, ge=1, le=12, description="Jumlah rekomendasi teratas (default Top-3)"),
) -> dict[str, Any]:
    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File bukan gambar yang valid: {exc}") from exc

    return run_prediction(image=image, berat_kg=berat_kg, top_k=top_k)


# --- Antarmuka Gradio (Gratis di Hugging Face Spaces SDK: Gradio + ZeroGPU) ---
@zerogpu
def predict_gradio(image: Image.Image | None, berat_kg: float) -> tuple[dict[str, float], dict[str, Any]]:
    if image is None:
        return {}, {"error": "Mohon unggah atau ambil foto sampah terlebih dahulu."}
    hasil = run_prediction(image=image, berat_kg=float(berat_kg or 1.0), top_k=3)
    label_scores = {
        f"{k['kategori']['nama_kategori']} ({k['label_model']})": round(k["confidence"], 4)
        for k in hasil["kandidat_teratas"]
    }
    return label_scores, hasil


demo = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Image(type="pil", label="Foto Sampah"),
        gr.Number(value=1.0, minimum=0.0, step=0.1, label="Berat Timbangan (kg)"),
    ],
    outputs=[
        gr.Label(num_top_classes=3, label="Top-3 Smart Suggestion"),
        gr.JSON(label="Detail Prediksi (JSON API)"),
    ],
    title="TRASHURY Cloud AI — SigLIP-SO400M",
    description="Cloud Inference Service untuk arsitektur Hybrid Cloud-Edge TRASHURY (DTETI FT UGM).",
    api_name="predict",
)

if __name__ == "__main__":
    _load_models()
    demo.launch(prevent_thread_lock=True)
    demo.app.add_api_route("/api/health", health_endpoint, methods=["GET"])
    demo.app.add_api_route("/api/predict", predict_endpoint, methods=["POST"])
    demo.block_thread()
