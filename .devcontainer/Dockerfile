FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System Tools
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip && pip install -r requirements.txt
RUN python3 -m pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

CMD ["bash"]
