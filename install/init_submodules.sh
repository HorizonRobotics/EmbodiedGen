#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"

usage() {
  echo "Usage: bash install/init_submodules.sh [basic|scene3d|room|affordance|cu126|all]"
}

case "$STAGE" in
  basic|scene3d|room|affordance|cu126|all) ;;
  *)
    usage >&2
    exit 1
    ;;
esac

submodules_for_stage() {
  case "$1" in
    basic)
      printf '%s\n' \
        "thirdparty/TRELLIS" \
        "thirdparty/sam3d"
      ;;
    scene3d)
      printf '%s\n' "thirdparty/pano2room"
      ;;
    room)
      printf '%s\n' "thirdparty/infinigen"
      ;;
    affordance)
      printf '%s\n' \
        "thirdparty/GraspGen" \
        "thirdparty/Hunyuan3D-Part"
      ;;
    all)
      git -C "$REPO_ROOT" config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{ print $2 }'
      ;;
    cu126)
      ;;
  esac
}

mapfile -t REQUESTED_SUBMODULES < <(submodules_for_stage "$STAGE")

echo "Repo root: $REPO_ROOT"
echo "Stage: $STAGE"

if [[ "${#REQUESTED_SUBMODULES[@]}" -eq 0 ]]; then
  echo "No submodules required for this stage."
  exit 0
fi

echo "Initializing submodules..."
HUNYUAN_PART_PATH="thirdparty/Hunyuan3D-Part"

submodule_known_to_git() {
  git -C "$REPO_ROOT" ls-files --error-unmatch "$1" >/dev/null 2>&1
}

submodule_target_commit() {
  local path="$1"
  git -C "$REPO_ROOT" rev-parse --verify "HEAD:$path" 2>/dev/null \
    || { echo "Missing submodule gitlink: $path" >&2; exit 1; }
}

init_submodule_checkout() {
  local path="$1"
  local recursive="${2:-0}"

  if ! submodule_known_to_git "$path"; then
    echo "Missing submodule gitlink: $path" >&2
    exit 1
  fi

  if [[ "$recursive" == "1" ]]; then
    git -C "$REPO_ROOT" submodule update --init --recursive --progress "$path"
  else
    git -C "$REPO_ROOT" submodule update --init --progress "$path"
  fi
}

while IFS= read -r path; do
  if ! git -C "$REPO_ROOT" config -f .gitmodules --get "submodule.$path.path" >/dev/null; then
    echo "Unknown submodule path in stage '$STAGE': $path" >&2
    exit 1
  fi

  if [[ "$path" == "$HUNYUAN_PART_PATH" && "${ASSET3D_HUNYUAN_FULL:-0}" != "1" ]]; then
    full_path="$REPO_ROOT/$path"
    commit="$(submodule_target_commit "$path")"
    url="$(git -C "$REPO_ROOT" config -f .gitmodules --get "submodule.$path.url")"
    branch="$(git -C "$REPO_ROOT" config -f .gitmodules --get "submodule.$path.branch" || true)"
    module_git_dir="$REPO_ROOT/.git/modules/$path"

    echo "Initializing $path with sparse checkout (skipping GLB assets)..."
    if submodule_known_to_git "$path"; then
      git -C "$REPO_ROOT" submodule init "$path"
    fi
    if [[ ! -d "$full_path/.git" && ! -f "$full_path/.git" ]]; then
      rm -rf "$full_path"
      if [[ ! -e "$module_git_dir" ]]; then
        mkdir -p "$(dirname "$module_git_dir")"
        git clone --separate-git-dir "$module_git_dir" --filter=blob:none --depth 1 --no-checkout --branch "${branch:-main}" "$url" "$full_path"
      else
        git clone --filter=blob:none --depth 1 --no-checkout --branch "${branch:-main}" "$url" "$full_path"
      fi
    fi
    git -C "$full_path" sparse-checkout init --no-cone
    git -C "$full_path" sparse-checkout set --no-cone '/*' '!*.glb'
    if ! git -C "$full_path" checkout --progress "$commit"; then
      git -C "$full_path" fetch --depth 1 origin "$commit"
      git -C "$full_path" checkout --progress "$commit"
    fi
    if [[ -d "$full_path/.git" && ! -e "$module_git_dir" ]]; then
      git -C "$REPO_ROOT" submodule absorbgitdirs "$path"
    elif [[ -d "$full_path/.git" ]]; then
      echo "Keeping standalone gitdir for $path; $module_git_dir already exists."
    fi
    continue
  fi

  if [[ "$path" == "$HUNYUAN_PART_PATH" && -e "$REPO_ROOT/$path/.git" ]]; then
    git -C "$REPO_ROOT/$path" sparse-checkout disable || true
  fi

  if [[ "$path" == "thirdparty/infinigen" ]]; then
    INFI_PATH="$REPO_ROOT/$path"
    echo "Initializing $path..."
    init_submodule_checkout "$path"

    if git -C "$INFI_PATH" rev-parse --verify ":src/infinigen/infinigen_gpl" >/dev/null 2>&1; then
      INFI_MODULE_ROOT="src/infinigen"
      INFI_EXAMPLES_ROOT="src/infinigen_examples"
    else
      INFI_MODULE_ROOT="infinigen"
      INFI_EXAMPLES_ROOT="infinigen_examples"
    fi

    echo "Recursively initializing infinigen_gpl..."
    git -C "$INFI_PATH" submodule update --init --recursive --progress "$INFI_MODULE_ROOT/infinigen_gpl"

    echo "Recursively initializing OcMesher..."
    git -C "$INFI_PATH" submodule update --init --recursive --progress "$INFI_MODULE_ROOT/OcMesher"

    echo "Recursively initializing glm..."
    git -C "$INFI_PATH" submodule update --init --recursive --progress "$INFI_MODULE_ROOT/datagen/customgt/dependencies/glm"

    echo "Applying patches to infinigen..."
    sed -i.bak \
      "s|'infinigen_examples|'thirdparty/infinigen/$INFI_EXAMPLES_ROOT|g" \
      "$INFI_PATH/$INFI_EXAMPLES_ROOT/configs_indoor/base_indoors.gin"
    rm -f "$INFI_PATH/$INFI_EXAMPLES_ROOT/configs_indoor/base_indoors.gin.bak"

    echo "Updating scikit-image constraint..."
    sed -i.bak \
      's|"scikit-image<0.20.0",|"scikit-image",|g' \
      "$INFI_PATH/pyproject.toml"
    rm -f "$INFI_PATH/pyproject.toml.bak"

    continue
  fi

  echo "Recursively initializing $path..."
  init_submodule_checkout "$path" 1

done < <(printf '%s\n' "${REQUESTED_SUBMODULES[@]}")
