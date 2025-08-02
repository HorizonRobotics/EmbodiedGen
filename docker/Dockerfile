FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
LABEL maintainer="Horizon Robotics <xinjie.wang@horizon.auto>" \
    version="0.1.2" \
    description="Docker image for EmbodiedGen"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" -> A100/4090/H100
ENV CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" \
    TCNN_CUDA_ARCHITECTURES=80,89,90

RUN useradd -m -s /bin/bash e_user
WORKDIR /EmbodiedGen
COPY . .
RUN chown -R e_user:e_user /EmbodiedGen
USER e_user

RUN conda create -n embodiedgen python=3.10.13 -y && \
    conda run -n embodiedgen bash install.sh

RUN /opt/conda/bin/conda init bash && \
    echo "conda activate embodiedgen" >> /home/e_user/.bashrc

CMD ["bash", "--login"]
