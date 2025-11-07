---
hide:
  - navigation
---

## ✅ Setup Environment
```sh
git clone https://github.com/HorizonRobotics/EmbodiedGen.git
cd EmbodiedGen
git checkout v0.1.6
git submodule update --init --recursive --progress
conda create -n embodiedgen python=3.10.13 -y # recommended to use a new env.
conda activate embodiedgen
bash install.sh basic
```

## ✅ Starting from Docker

We provide a pre-built Docker image on [Docker Hub](https://hub.docker.com/repository/docker/wangxinjie/embodiedgen) with a configured environment for your convenience. For more details, please refer to [Docker documentation](https://github.com/HorizonRobotics/EmbodiedGen/tree/master/docker).

> **Note:** Model checkpoints are not included in the image, they will be automatically downloaded on first run. You still need to set up the GPT Agent manually.

```sh
IMAGE=wangxinjie/embodiedgen:env_v0.1.x
CONTAINER=EmbodiedGen-docker-${USER}
docker pull ${IMAGE}
docker run -itd --shm-size="64g" --gpus all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged --net=host --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} bash
```

## ✅ Setup GPT Agent

Update the API key in file: `embodied_gen/utils/gpt_config.yaml`.

You can choose between two backends for the GPT agent:

- **`gpt-4o`** (Recommended) – Use this if you have access to **Azure OpenAI**.
- **`qwen2.5-vl`** – An alternative with free usage via OpenRouter, apply a free key [here](https://openrouter.ai/settings/keys) and update `api_key` in `embodied_gen/utils/gpt_config.yaml` (50 free requests per day)
