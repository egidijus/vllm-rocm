# syntax=docker/dockerfile:1-labs
ARG AMDGPU_FAMILY=gfx120X-all
ARG GPU_ARCH=gfx1201
ARG ROCM_VERSION=7.13.0a20260327
#ARG ROCM_VERSION=7.12.0a20260311
#ARG ROCM_VERSION=7.12.0a20260205
ARG VLLM_BRANCH="releases/v0.18.1"


FROM ubuntu:24.04 AS base
ENV PYTHONUNBUFFERED=1
ARG AMDGPU_FAMILY
ARG GPU_ARCH
ARG ROCM_VERSION
ARG VLLM_BRANCH

# DOC Can't self reference $PATH becayse $PATH does not exist before an image is "FROMMED"

ENV ROCM_PATH=/opt/rocm
ENV PATH="/root/.local/bin:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:/usr/sbin:/sbin:/bin/usr:/bin:${PATH}"

SHELL ["/bin/bash", "-l", "-c"]
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libatomic1 \
    libgomp1 \
    libnuma-dev \
    kmod \
    wget && \
    rm -rf /var/lib/apt/lists/*

# setup venv and make the env active for all shell sessions,
# including run commands
# 2026-03-27 https://github.com/vllm-project/vllm/commit/af89140efc06c462ae531999b9f2db6ba0c7a528
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
RUN curl -LsSf --retry 3 --retry-delay 5 https://astral.sh/uv/install.sh -o /tmp/uv-install.sh \
    && env UV_INSTALL_DIR="/root/.local/bin" sh /tmp/uv-install.sh \
    && rm -f /tmp/uv-install.sh \
    && uv --version
# ENV PATH="/root/.local/bin:${PATH}"
RUN cd /app && uv venv --python 3.12 && \
    source .venv/bin/activate && \
    uv pip install pip && \
    echo "source /app/.venv/bin/activate" > /root/.bash_profile

RUN /sbin/modprobe amdgpu

# install ROCm python packages
RUN uv pip install \
    --index-url https://rocm.nightlies.amd.com/v2/${AMDGPU_FAMILY}/ \
    "rocm[libraries, devel]" && \
    uv pip install \
    --index-url https://rocm.nightlies.amd.com/v2/${AMDGPU_FAMILY}/ \
    torch torchvision torchaudio

# install tarball of rocm
RUN mkdir -p /opt/rocm-$ROCM_VERSION && \
    wget https://rocm.nightlies.amd.com/tarball/therock-dist-linux-$AMDGPU_FAMILY-$ROCM_VERSION.tar.gz && \
    tar xzf ./therock-dist-linux-$AMDGPU_FAMILY-$ROCM_VERSION.tar.gz -C /opt/rocm-$ROCM_VERSION && \
    rm therock-dist-linux-$AMDGPU_FAMILY-$ROCM_VERSION.tar.gz && \
    ln -s /opt/rocm-$ROCM_VERSION /opt/rocm


ENV LD_LIBRARY_PATH=$ROCM_PATH/lib
# ENV CMAKE_PREFIX_PATH="/app/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch"
ENV DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode  
ENV HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}

ENV CC=$ROCM_PATH/llvm/bin/clang
ENV CXX=$ROCM_PATH/llvm/bin/clang++
ENV HIPCC=$ROCM_PATH/bin/hipcc
ENV VLLM_TARGET_DEVICE="rocm"
ENV GPU_TARGETS="${GPU_ARCH}"
ENV Torch_DIR="/app/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch"

# copy .bash_profile to .bashrc
RUN cp /root/.bash_profile /root/.bashrc



# clone vllm
RUN --security=insecure git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout -b ${VLLM_BRANCH} && \
    python use_existing_torch.py && \
    uv pip install --upgrade numba \
        scipy \
        cmake \
        setuptools_scm && \
    uv pip install "numpy<2" && \
    # uv pip install -r requirements/rocm.txt
    uv pip install -r requirements/rocm.txt && \
    python3 setup.py clean --all  && \
    python3 setup.py bdist_wheel --dist-dir=dist && \
    uv pip install /opt/rocm/share/amd_smi

# DOC check if amd-smi works, can we detect gpu
RUN $ROCM_PATH/bin/amd-smi list

# Verify that PyTorch is the ROCm build, not CUDA
RUN python3 -c "import torch; assert torch.version.hip is not None, \
    f'Expected ROCm PyTorch but got CUDA (torch.version.cuda={torch.version.cuda}, torch.version.hip={torch.version.hip})'; \
    print(f'Verified: PyTorch {torch.__version__} with ROCm (HIP {torch.version.hip})')"

# RUN git clone https://github.com/hyoon1/flash-attention.git && \    
# RUN git clone https://github.com/ROCm/flash-attention.git && \
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    # git checkout enable-ck-gfx12 && \
    FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

ENTRYPOINT [ "/app/.venv/bin/vllm","serve"]
