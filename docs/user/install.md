Installation
=====================

### How to install

#### System requirements

- Nvidia gpu (GTX 10x0 or later, earlier may work but not tested)
- OS: Linux (We prefer Ubuntu 16.04)

We provide the `Dockerfile` and `docker-compose.yml` for you to install the enviroment. First we need to install docker in host machine

#### Install Docker 

1. Install Docker

    Install Docker from [Homepage]("https://docs.docker.com/install/")
    
2. Install CUDA on Host Machine
    
    [Install Cuda 8.0 on Ubuntu 16.04]("https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04")

3. Install docker-compose
    
    [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

4. Install Nvidia-docker
    
    In order to passthrough GPU to docker we need to install [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

#### Create Docker image

1. Use docker-compose to create a docker image

    ```bash
        cd ~\workspace\faster-rcnn
        docker-compose up --build
    ```

    
