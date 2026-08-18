FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

LABEL maintainer="Horizon Robotics" \
      description="EmbodiedGen for RTX 5090 (sm_120, CUDA 12.8)"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build pkg-config \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
        libegl1 libgles2 libgbm1 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    EMBODIEDGEN_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    FORCE_CUDA=1 \
    MAX_JOBS=8

ENV HF_HOME=/hf_cache \
    TORCH_HOME=/hf_cache/torch_hub \
    OPENCV_IO_ENABLE_OPENEXR=1

WORKDIR /EmbodiedGen
COPY . /EmbodiedGen/

RUN conda create -n embodiedgen python=3.10.13 -y && \
    conda run -n embodiedgen bash install/install_basic.sh && \
    conda run -n embodiedgen python -m pip install numpy==1.26.4

RUN /opt/conda/bin/conda init bash && \
    echo "conda activate embodiedgen" >> /root/.bashrc

CMD ["bash", "--login"]
