---
hide:
  - navigation
---

## ✅ Setup Environment
```sh
git clone https://github.com/HorizonRobotics/EmbodiedGen.git
cd EmbodiedGen
git checkout v2.1.0
conda create -n embodiedgen python=3.10.13 -y # recommended to use a new env.
conda activate embodiedgen
# Manually install one CUDA toolkit when needed. cu126 remains the default.
# bash install.sh cu126 # RTX 40-series.
# bash install.sh cu128 # RTX 50-series / Blackwell.
conda deactivate && conda activate embodiedgen
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

EmbodiedGen supports three GPT agent backends:

- **Azure OpenAI** for managed Azure deployments.
- **OpenRouter** for OpenAI-compatible hosted models.
- **Codex CLI** for local developers who already use `codex login`.

Azure OpenAI and OpenRouter require an API key in
`embodied_gen/utils/gpt_config.yaml`. Codex uses the local Codex CLI login and
does not require storing a key in the project configuration. See
[GPT Agent Setup](gpt_agent.md) for complete configuration and deployment
notes.
