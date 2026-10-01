FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="PI-Ensemble" \
      org.opencontainers.image.description="PI-Ensemble CUDA 12.8 runtime with Boltz, ESM, ProteinMPNN, OpenMM, and cg2all"

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    APP_NAME=PI-Ensemble \
    PYTHONUNBUFFERED=1

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        sudo && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniforge.sh && \
    conda config --system --set channel_priority strict && \
    conda clean -afy

RUN conda create -y -n pie python=3.12 pip setuptools wheel && \
    conda create -y -n bioemu python=3.11 pip setuptools wheel && \
    conda create -y -n cg2all python=3.11 pip setuptools wheel && \
    conda clean -afy

ENV PATH="${CONDA_DIR}/envs/pie/bin:${CONDA_DIR}/bin:${PATH}" \
    CONDA_DEFAULT_ENV=pie \
    PROTEINMPNN_HOME=/opt/ProteinMPNN \
    DGLBACKEND=pytorch

RUN git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN && \
    git -C /opt/ProteinMPNN checkout 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57

WORKDIR /opt/PI-Ensemble
COPY . /opt/PI-Ensemble

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        torch==2.7.1 \
        torchvision==0.22.1 && \
    python -m pip install --no-cache-dir \
        "numpy<2" \
        "mdtraj<1.11" && \
    python -m pip install --no-cache-dir . && \
    python -m pip install --no-cache-dir \
        pyyaml \
        requests \
        httpx \
        pandas \
        python-Levenshtein \
        openmm \
        boltz==2.2.0 \
        "esm @ git+https://github.com/Biohub/esm.git@main" \
        cuequivariance==0.6.0 \
        cuequivariance-torch==0.6.0 \
        cuequivariance-ops-cu12==0.6.0 \
        cuequivariance-ops-torch-cu12==0.6.0

RUN eval "$(conda shell.bash hook)" && \
    conda activate bioemu && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        torch==2.7.1 \
        torchvision==0.22.1 && \
    python -m pip install --no-cache-dir bioemu==1.3.1 && \
    conda clean -afy

RUN eval "$(conda shell.bash hook)" && \
    conda activate cg2all && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir git+https://github.com/huhlim/cg2all.git@a00b8816736c08852944f147e39164d0f5e1834e && \
    python -m pip install --no-cache-dir e3nn==0.5.1 && \
    conda clean -afy

CMD ["bash"]
