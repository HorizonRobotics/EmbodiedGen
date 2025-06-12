#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting installation process...${NC}"
git config --global http.postBuffer 524288000

echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt --use-deprecated=legacy-resolver --default-timeout=60 || {
    echo -e "${RED}Failed to install requirements${NC}"
    exit 1
}


echo -e "${GREEN}Installing kaolin from GitHub...${NC}"
pip install kaolin@git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.16.0 || {
    echo -e "${RED}Failed to install kaolin${NC}"
    exit 1
}


echo -e "${GREEN}Installing flash-attn...${NC}"
pip install flash-attn==2.7.0.post2 --no-build-isolation || {
    echo -e "${RED}Failed to install flash-attn${NC}"
    exit 1
}


echo -e "${GREEN}Installing diff-gaussian-rasterization...${NC}"
TMP_DIR="/tmp/mip-splatting"
rm -rf "$TMP_DIR"
git clone --recursive https://github.com/autonomousvision/mip-splatting.git "$TMP_DIR" && \
pip install "$TMP_DIR/submodules/diff-gaussian-rasterization" && \
rm -rf "$TMP_DIR" || {
    echo -e "${RED}Failed to clone or install diff-gaussian-rasterization${NC}"
    rm -rf "$TMP_DIR"
    exit 1
}
echo -e "${GREEN}Installation completed successfully!${NC}"


echo -e "${GREEN}Installing gsplat from GitHub...${NC}"
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.5.0 || {
    echo -e "${RED}Failed to install gsplat${NC}"
    exit 1
}


echo -e "${GREEN}Installing EmbodiedGen...${NC}"
pip install -e . || {
    echo -e "${RED}Failed to install local package${NC}"
    exit 1
}

echo -e "${GREEN}Installation completed successfully!${NC}"

