This directory contains everything needed to create a `gunpowder` Docker image.

## Setup nvidia-docker

```bash
# nvidia drivers
sudo apt-get install nvidia-375
sudo apt-get install libcuda1-375

# install docker
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo docker run hello-world
sudo apt-get install nvidia-docker

# install nvidia-docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# start nvidia-docker (if failed above)
sudo service nvidia-docker start
sudo nvidia-docker run --rm nvidia/cuda nvidida-smi
```

## Build image

```bash
sudo nvidia-docker build -t gunpowder .
```

## Run image as container

```bash
sudo nvidia-docker run gunpowder
```
