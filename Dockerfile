FROM python:3.11-slim

# Non-root user wajib untuk Hugging Face Spaces Docker SDK
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/tmp/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install PyTorch CPU-only agar image jauh lebih ringan & cepat build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user app.py .
COPY --chown=user models/siglip_ens_mlp2_head.onnx models/siglip_ens_mlp2_head.onnx
COPY --chown=user models/labels.txt models/labels.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
