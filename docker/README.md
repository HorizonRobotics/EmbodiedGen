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

**Note**: Model checkpoints are not included in the image (auto-download on the first run), and you still need to configure the GPT agent. Refer to the [Setup GPT Agent](https://github.com/HorizonRobotics/EmbodiedGen?tab=readme-ov-file#-setup-gpt-agent) section for detailed instructions.


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
