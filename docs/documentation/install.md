---
hide:
  - navigation
---

## ✅ Setup Environment
```sh
git clone https://github.com/HorizonRobotics/EmbodiedGen.git
cd EmbodiedGen
git checkout v2.0.0
conda create -n embodiedgen python=3.10.13 -y # recommended to use a new env.
conda activate embodiedgen
# bash install.sh cu126 && conda deactivate && conda activate embodiedgen # Optional: if you don't have local cuda126.
bash install.sh basic # around 10 mins
# Optional: `bash install.sh scene3d` for scene3d-cli; `bash install.sh room` for room-cli; `bash install.sh affordance` for affordance-cli.
```

Please `huggingface-cli login` to ensure that the ckpts can be downloaded automatically afterwards.

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

- **`gpt-5.4`** (or Higher, Recommended) – Use this if you have access to **Azure OpenAI**.
- **`gemma-4-31b`** – A free multimodal alternative (`google/gemma-4-31b-it:free`) via OpenRouter, apply a free key [here](https://openrouter.ai/settings/keys) and update `api_key` in `embodied_gen/utils/gpt_config.yaml`. Free-tier availability on OpenRouter can change over time — check the [model page](https://openrouter.ai/google/gemma-4-31b-it:free) before relying on it.
