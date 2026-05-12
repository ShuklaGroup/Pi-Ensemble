FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
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

RUN conda create -y -n pie python=3.11 pip setuptools wheel && \
    conda create -y -n cg2all python=3.11 pip setuptools wheel && \
    conda clean -afy

ENV PATH="${CONDA_DIR}/envs/pie/bin:${CONDA_DIR}/bin:${PATH}" \
    CONDA_DEFAULT_ENV=pie \
    PROTEINMPNN_HOME=/opt/ProteinMPNN \
    DGLBACKEND=pytorch

RUN git clone https://github.com/Kuhlman-Lab/proteinmpnn.git /opt/ProteinMPNN && \
    ln -s /opt/ProteinMPNN/run/protein_mpnn/protein_mpnn_run.py /opt/ProteinMPNN/protein_mpnn_run.py

WORKDIR /opt/pie
COPY . /opt/pie

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.6.0 \
        torchvision==0.21.0 && \
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
        boltz==2.0.3 \
        esm

RUN eval "$(conda shell.bash hook)" && \
    conda activate cg2all && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir git+https://github.com/huhlim/cg2all.git && \
    python -m pip install --no-cache-dir e3nn==0.5.1 && \
    conda clean -afy

CMD ["bash"]
