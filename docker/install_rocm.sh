#!/bin/bash
# EmbodiedGen ROCm install (gfx942 / MI300X / MI308X), ROCm 6.4.3 + PyTorch 2.6.
# Swaps the CUDA-only generation stack for ROCm-compatible builds following the
# rocm-lib-compat reference (github.com/ZJLi2013/rocm3d). Intended to run inside
# rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 with the repo at
# /workspace/EmbodiedGen. Each step reports PASS/FAIL but does not abort, so one
# run yields a full ROCm-compat status map.
set -uo pipefail

export PYTORCH_ROCM_ARCH=gfx942
export GPU_ARCHS=gfx942
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PIP_ROOT_USER_ACTION=ignore
REPO=${REPO:-/workspace/EmbodiedGen}
cd "$REPO"

PASS=(); FAIL=()
step () {  # step "name" cmd...
  local name="$1"; shift
  echo "==================== STEP: $name ===================="
  if "$@"; then echo "[PASS] $name"; PASS+=("$name");
  else echo "[FAIL] $name"; FAIL+=("$name"); fi
}
pipi () { pip install --no-cache-dir "$@"; }

# --- 0. keep ROCm torch from base image (do NOT reinstall cu118 torch) ---
python -c "import torch;print('base torch',torch.__version__,'hip',torch.version.hip,'gpu',torch.cuda.is_available())"

# --- 1. requirements.txt minus CUDA-pinned libs (handled below or via base) ---
EXCLUDE='torch|torchvision|torchaudio|xformers|gsplat|flash.attn|flash-attn|triton|spconv|spconv-cu120|pytorch3d'
grep -vEi "^(${EXCLUDE})([<>=!~;[:space:]]|$)" requirements.txt > /tmp/req_clean.txt
echo "--- cleaned requirements (CUDA libs stripped) ---"; cat /tmp/req_clean.txt
step "requirements(clean)" pipi -r /tmp/req_clean.txt --use-deprecated=legacy-resolver
# numpy: EmbodiedGen's diffusers/transformers REQUIRE numpy<2. (The rocm-lib-compat
# "use docker numpy 2.x" guidance does NOT apply here; the base image ships numpy 2.x.)
step "numpy<2" pipi "numpy<2"
# NOTE: xformers is NOT required -- SAM3D attention auto-selects the `sdpa` backend on
# ROCm, and TRELLIS uses spconv+flash_attn. Skipping xformers avoids the torch 2.9.1
# bump that would break the pytorch3d/gsplat ROCm 6.4 wheels.

# --- 2. ROCm replacements for the CUDA-only generation stack (rocm-lib-compat) ---
#     All verified on gfx942 / MI300X, ROCm 6.4 (same stack as VAST-AI/AniGen,
#     a TRELLIS-v1 image-to-3D repo, in the rocm3d supported-repo list).
# spconv  (CUDA spconv-cu120 -> ZJLi2013/spconv_rocm)
step "spconv_rocm" bash -c '
  rm -rf /tmp/spconv_rocm &&
  git clone --depth 1 -b rocm https://github.com/ZJLi2013/spconv_rocm.git /tmp/spconv_rocm &&
  pip install --no-cache-dir -e /tmp/spconv_rocm'
# nvdiffrast (NVlabs -> ZJLi2013/nvdiffrast@rocm)
step "nvdiffrast_rocm" pipi "git+https://github.com/ZJLi2013/nvdiffrast.git@rocm" --no-build-isolation
# gsplat (-> amd_gsplat prebuilt; import name stays `gsplat`; default gaussian backend)
step "amd_gsplat" pipi amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/
# pytorch3d (-> prebuilt ROCm 6.4 / py3.12 wheel)
step "pytorch3d_rocm" pipi https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl
# flash-attn (FA2 Triton on ROCm 6.4). NOTE: requires FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# at BOTH install and runtime; otherwise import falls back to the CUDA `flash_attn_2_cuda`.
step "flash_attn(triton)" bash -c 'FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-cache-dir flash-attn --no-build-isolation'

# --- 3. pure git deps from upstream install_basic.sh (CUDA-agnostic) ---
step "utils3d"  pipi "utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
step "clip"     pipi "clip@git+https://github.com/openai/CLIP.git"
step "segment_anything" pipi "segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f"
step "kolors"   pipi "kolors@git+https://github.com/HochCC/Kolors.git"
step "MoGe"     pipi "MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734"

# --- 4. OPTIONAL: diff-gaussian-rasterization ('inria' gaussian backend) ---
# img3d's default gaussian backend is gsplat (amd_gsplat, above), and both TRELLIS
# and SAM3D guard the diff_gaussian_rasterization import in try/except, so this is
# optional. The CUDA-clean ROCm source is graphdeco-inria built via expenses/
# gaussian-splatting's ROCm branch (rocm3d supported-repo list). EmbodiedGen wires
# TRELLIS to the *mip-splatting* antialiasing fork, whose ROCm build additionally
# needs `__trap`->abort and a cooperative_groups/reduce.h shim (PR candidate).
# Left out of the default install; uncomment to add the non-AA 'inria' backend:
#   touch /opt/rocm/include/device_launch_parameters.h
#   step "diff_gaussian_rasterization" bash -c '
#     rm -rf /tmp/dgr &&
#     git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization /tmp/dgr &&
#     cd /tmp/dgr && git submodule update --init --recursive &&
#     PYTORCH_ROCM_ARCH=gfx942 pip install --no-cache-dir . --no-build-isolation'

# --- 5. ROCm runtime shims, installed as sitecustomize (run at interpreter startup) ---
# (a) kaolin bypass: kaolin is CUDA-only (no ROCm wheel, setup.py hard-requires nvcc).
#     Imported at module top of embodied_gen/data/utils.py but only *used* inside the
#     texture-backprojection / mesh-IO stage (kal.io.*.import_mesh, render.materials,
#     render.camera) + type refs in thirdparty/sam3d -- none on the core image->3D
#     geometry+gaussian path. Stub pattern proven on ZJLi2013/RealWonder (same
#     SAM-3D-Objects/kaolin dep). check_tensor-style validators return truthy.
# (b) spconv KRSC->Native weight bridge: SAM3D/TRELLIS checkpoints store sparse-conv
#     weights in CUDA spconv's ImplicitGemm KRSC layout (5D [out,k,k,k,in]); spconv_rocm
#     falls back to the Native algo (3D [Kvol,in,out], 2D when kvol==1), so load_state_dict
#     mismatches. The shim converts on load. (Upstream fix belongs in spconv_rocm.)
SITE=$(python -c "import site;print(site.getsitepackages()[0])")
HERE="$(dirname "$0")"
if cp "$HERE/kaolin_stub.py" "$SITE/kaolin_stub.py" \
   && cp "$HERE/spconv_rocm_compat.py" "$SITE/spconv_rocm_compat.py" \
   && printf 'import kaolin_stub\nimport spconv_rocm_compat\n' > "$SITE/sitecustomize.py"; then
  echo "[PASS] rocm-shims -> $SITE/sitecustomize.py (kaolin_stub + spconv_rocm_compat)"; PASS+=("rocm-shims")
else
  echo "[FAIL] rocm-shims copy"; FAIL+=("rocm-shims")
fi

# --- 6. import smoke: what actually loads on ROCm (flash-attn needs the env var) ---
echo "==================== IMPORT SMOKE ===================="
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python - <<'PY'
core = ["torch","spconv","nvdiffrast.torch","gsplat","pytorch3d","flash_attn","trimesh","diffusers"]
optional = ["diff_gaussian_rasterization","kaolin"]
import importlib
def check(m):
    try:
        importlib.import_module(m); print(f"[import OK ] {m}")
    except Exception as e:
        print(f"[import ERR] {m}: {type(e).__name__}: {str(e)[:160]}")
print("-- core --");   [check(m) for m in core]
print("-- optional --"); [check(m) for m in optional]
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "gpu", torch.cuda.is_available())
PY

echo "==================== SUMMARY ===================="
echo "PASS (${#PASS[@]}): ${PASS[*]}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]}"
echo "INSTALL_ROCM_DONE"
