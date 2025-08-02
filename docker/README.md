# Docker

## Getting Started with Our Pre-built Docker Image

We provide pre-built Docker image on [Docker Hub](https://hub.docker.com/repository/docker/wangxinjie/embodiedgen) that includes a configured environment for your convenience.

```sh
IMAGE=wangxinjie/embodiedgen:env_v0.1.x
CONTAINER=EmbodiedGen-docker-${USER}
docker pull ${IMAGE}
docker run -itd --shm-size="64g" --gpus all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged --net=host --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} bash
# ref `EmbodiedGen/README.md` to get start.
```

> **Note**: Model checkpoints are not included in the default image, they will be automatically downloaded on first run. Also, you still need to configure the GPT agent manually. See the [Setup GPT Agent](https://github.com/HorizonRobotics/EmbodiedGen?tab=readme-ov-file#-setup-gpt-agent) section for detailed instructions.

If you prefer an image with all model checkpoints, you can use `wangxinjie/embodiedgen:v0.1.x`. However, please note that this image is significantly larger. We recommend using the lighter image and allowing the models to download on demand.


## Getting Started with Building from the Dockerfile
You can also build your customized docker based on our Dockerfile.

```sh
git clone https://github.com/HorizonRobotics/EmbodiedGen.git
cd EmbodiedGen
TAG=v0.1.2 # Change to the latest stable version.
git checkout $TAG
git submodule update --init --recursive --progress

docker build -t embodiedgen:$TAG -f docker/Dockerfile .
```
