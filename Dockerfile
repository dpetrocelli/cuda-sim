# GPGPU-Sim 4.2.1 (Feb 2025) sobre CUDA 11.7 / Ubuntu 20.04
# Simulador cycle-accurate de GPUs NVIDIA — corre kernels CUDA sin placa fisica.
# Base oficial NVIDIA. CUDA 11.7 es el sweet spot probado para GPGPU-Sim 4.x.
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Argentina/Buenos_Aires

# Dependencias de build de GPGPU-Sim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    xutils-dev \
    bison \
    zlib1g-dev \
    flex \
    libglu1-mesa-dev \
    doxygen \
    graphviz \
    python3 \
    python3-pip \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Clonamos y compilamos GPGPU-Sim (rama master = 4.2.1 al dia de hoy)
WORKDIR /opt
RUN git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
WORKDIR /opt/gpgpu-sim_distribution

# Build
RUN bash -c "source setup_environment && make -j$(nproc)"

# Script de entrada que carga el entorno de GPGPU-Sim automaticamente
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'cd /opt/gpgpu-sim_distribution && source setup_environment > /dev/null 2>&1' >> /entrypoint.sh && \
    echo 'cd /workspace' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
