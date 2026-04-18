FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + deadsnakes PPA for Python 3.12 (not in Jammy default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Layer 1: Install torch with CUDA 12.6 (large, cached separately)
RUN pip install --no-cache-dir \
    torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Layer 2: Install remaining Python dependencies
# NOTE: discrete-speech-metrics (SpeechBERTScore) is excluded — pypesq build is
# broken with modern numpy/setuptools. Not needed for Phase 3 (--skip-speechbertscore).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Layer 4: Pre-download models (baked into image, avoids runtime downloads)
ENV HF_HOME=/opt/models/huggingface
ENV TORCH_HOME=/opt/models/torch

RUN python3 -c "\
from qwen_tts import Qwen3TTSModel; \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu')"

RUN python3 -c "\
import torch; \
torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)"

RUN python3 -c "\
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector; \
Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv'); \
WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')"

RUN python3 -c "\
import whisper; whisper.load_model('turbo')"

RUN python3 -c "\
from transformers import WavLMModel; \
WavLMModel.from_pretrained('microsoft/wavlm-large')"

# Copy project code and data
WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/
COPY data/libritts_r_aligned/ data/libritts_r_aligned/

# Default entrypoint runs the experiment
ENTRYPOINT ["python3", "scripts/run_fewshot.py"]

# Default args: Phase 3 full run
CMD [ \
    "--manifest", "data/libritts_r_aligned/manifest.json", \
    "--approaches", "single_baseline", "embed_avg", "concat_audio", \
    "--num-refs", "1", "2", "3", "5", \
    "--strategies", "random", "longest", \
    "--seeds", "42", "123", "456", \
    "--held-out-targets", "5", \
    "--device", "cuda:0", \
    "--dtype", "float16", \
    "--flash-attn", \
    "--skip-speechbertscore" \
]
