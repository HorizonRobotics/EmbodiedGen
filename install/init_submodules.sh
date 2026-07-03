#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Repo root: $REPO_ROOT"

echo "Initializing submodules..."
HUNYUAN_PART_PATH="thirdparty/Hunyuan3D-Part"

while IFS= read -r path; do
  if [[ "$path" == "$HUNYUAN_PART_PATH" && "${ASSET3D_HUNYUAN_FULL:-0}" != "1" ]]; then
    full_path="$REPO_ROOT/$path"
    commit="$(git -C "$REPO_ROOT" rev-parse "HEAD:$path")"
    url="$(git -C "$REPO_ROOT" config -f .gitmodules --get "submodule.$path.url")"
    branch="$(git -C "$REPO_ROOT" config -f .gitmodules --get "submodule.$path.branch" || true)"

    echo "Initializing $path with sparse checkout (skipping GLB assets)..."
    git -C "$REPO_ROOT" submodule init "$path"
    if [[ ! -d "$full_path/.git" && ! -f "$full_path/.git" ]]; then
      rm -rf "$full_path"
      git clone --filter=blob:none --depth 1 --no-checkout --branch "${branch:-main}" "$url" "$full_path"
    fi
    git -C "$full_path" sparse-checkout init --no-cone
    git -C "$full_path" sparse-checkout set --no-cone '/*' '!*.glb'
    if ! git -C "$full_path" checkout --progress "$commit"; then
      git -C "$full_path" fetch --depth 1 origin "$commit"
      git -C "$full_path" checkout --progress "$commit"
    fi
    git -C "$REPO_ROOT" submodule absorbgitdirs "$path"
    continue
  fi

  if [[ "$path" == "$HUNYUAN_PART_PATH" && -e "$REPO_ROOT/$path/.git" ]]; then
    git -C "$REPO_ROOT/$path" sparse-checkout disable || true
  fi

  if [[ "$path" == "thirdparty/infinigen" ]]; then
    INFI_PATH="$REPO_ROOT/$path"
    echo "Initializing $path..."
    git -C "$REPO_ROOT" submodule update --init --progress "$path"

    echo "Filtering .gitmodules in $INFI_PATH..."
    git -C "$INFI_PATH" config -f .gitmodules --get-regexp '^submodule\..*\.path$' \
    | while read -r key value; do
        name=$(echo "$key" | sed 's/^submodule\.//;s/\.path$//')
        if [[ "$name" != "infinigen/infinigen_gpl" \
           && "$name" != "infinigen/OcMesher" \
           && "$name" != "infinigen/datagen/customgt/dependencies/glm" ]]; then
          git -C "$INFI_PATH" config -f .gitmodules --remove-section "submodule.$name" || true
        fi
      done

    echo "Recursively initializing infinigen_gpl..."
    git -C "$INFI_PATH" submodule update --init --recursive --progress infinigen/infinigen_gpl

    echo "Recursively initializing OcMesher..."
    git -C "$INFI_PATH" submodule update --init --recursive --progress infinigen/OcMesher

    echo "Recursively initializing glm..."
    git -C "$INFI_PATH" submodule update --init --recursive --progress infinigen/datagen/customgt/dependencies/glm

    echo "Applying patches to infinigen..."
    sed -i.bak \
      "s|'infinigen_examples|'thirdparty/infinigen/infinigen_examples|g" \
      "$INFI_PATH/infinigen_examples/configs_indoor/base_indoors.gin"
    rm -f "$INFI_PATH/infinigen_examples/configs_indoor/base_indoors.gin.bak"

    echo "Updating scikit-image constraint..."
    sed -i.bak \
      's|"scikit-image<0.20.0",|"scikit-image",|g' \
      "$INFI_PATH/pyproject.toml"
    rm -f "$INFI_PATH/pyproject.toml.bak"

    continue
  fi

  echo "Recursively initializing $path..."
  git -C "$REPO_ROOT" submodule update --init --recursive --progress "$path"

done < <(git -C "$REPO_ROOT" config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{ print $2 }')
