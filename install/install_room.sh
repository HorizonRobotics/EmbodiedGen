#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
INFINIGEN_DIR="${PROJECT_ROOT}/thirdparty/infinigen"

BLENDER_WGET_LINK='https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz'
BLENDER_WGET_FILE="${INFINIGEN_DIR}/blender-4.2.0-linux-x64.tar.xz"
BLENDER_UNTAR_DIR_NAME='blender-4.2.0-linux-x64'
BLENDER_UNTAR_PATH="${INFINIGEN_DIR}/${BLENDER_UNTAR_DIR_NAME}"
BLENDER_DIR="${INFINIGEN_DIR}/blender"
BLENDER_PYTHON="${BLENDER_DIR}/4.2/python/bin/python3.11"

if [ ! -d "${BLENDER_DIR}" ]; then
    echo "Blender not found in ${BLENDER_DIR}, install..."

    # Download Blender to specific path
    if [ ! -f "${BLENDER_WGET_FILE}" ]; then
        wget -O "${BLENDER_WGET_FILE}" "${BLENDER_WGET_LINK}"
    fi

    # Unzip Blender in Linux
    tar -xf "${BLENDER_WGET_FILE}" -C "${INFINIGEN_DIR}"
    mv "${BLENDER_UNTAR_PATH}" "${BLENDER_DIR}"
    rm "${BLENDER_WGET_FILE}"
fi


ENV_NAME="build_311"
if ! conda info --envs | grep -q "^$ENV_NAME "; then
    conda create -n "$ENV_NAME" python=3.11 -y
    conda install -n "$ENV_NAME" -y -c conda-forge gxx_linux-64=11 mesalib glew glm menpo::glfw3
fi

# Set environment variables
TARGET_ENV=$(conda info --envs | awk -v env="$ENV_NAME" '$1 == env {print $NF; exit}')
if [ -z "$TARGET_ENV" ]; then
    CONDA_BASE=$(conda info --base)
    TARGET_ENV="$CONDA_BASE/envs/$ENV_NAME"
fi
export C_INCLUDE_PATH=$TARGET_ENV/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$TARGET_ENV/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$TARGET_ENV/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$TARGET_ENV/lib:$LD_LIBRARY_PATH
export CC=$TARGET_ENV/bin/x86_64-conda-linux-gnu-gcc
export CXX=$TARGET_ENV/bin/x86_64-conda-linux-gnu-g++

CUDA_HOST_CXX=
if [ -x "$TARGET_ENV/bin/x86_64-conda-linux-gnu-g++" ]; then
    CUDA_HOST_CXX="$TARGET_ENV/bin/x86_64-conda-linux-gnu-g++"
elif [ -x /usr/bin/g++-11 ]; then
    CUDA_HOST_CXX=/usr/bin/g++-11
fi
if [ -n "$CUDA_HOST_CXX" ]; then
    export CUDAHOSTCXX="$CUDA_HOST_CXX"
    export NVCC_PREPEND_FLAGS="-ccbin=$CUDA_HOST_CXX"
fi

"${BLENDER_PYTHON}" -m ensurepip
BUILD_PYTHONPATH="$TARGET_ENV/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
CFLAGS="-I$TARGET_ENV/include/python3.11 $CFLAGS" PYTHONPATH="$BUILD_PYTHONPATH" "${BLENDER_PYTHON}" -m pip install --no-build-isolation -e "${INFINIGEN_DIR}[sim]"

"${BLENDER_PYTHON}" -m pip install pyyaml tyro colorlog openai tenacity json-repair packaging
"${BLENDER_PYTHON}" -m pip install -e "${PROJECT_ROOT}"

echo "Setup room env complete."
