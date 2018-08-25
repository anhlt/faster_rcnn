Installation
============


#### System requirements

- Nvidia gpu (GTX 10x0 or later, earlier may work but not tested)
- OS: Linux (We prefer Ubuntu 16.04)

We provide the `Dockerfile` and `docker-compose.yml` for you to install the enviroment. First we need to install docker in host machine

#### Install Docker 

1. Install Docker

    Install Docker from [Homepage](https://docs.docker.com/install/)
    
2. Install CUDA on Host Machine
    
    [Install Cuda 8.0 on Ubuntu 16.04](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)

3. Install docker-compose
    
    [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

4. Install Nvidia-docker
    
    In order to passthrough GPU to docker we need to install [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

#### Create Docker image

1. Clone from github:

    ```bash
    git clone git@github.com:anhlt/faster_rcnn.git
    ```

2. Use docker-compose to create a docker image

    ```bash
    cd ~/workspace/faster_rcnn
    docker-compose up --build
    ```


#### Compile Cython module 

There are 3 modules need to be compiled, `nms`, `roi_pooling`, `utils`. We need to exec `\bin\bash` on Docker image to build those modules


```bash
    cd ~/workspace/faster_rcnn
    docker-compose exec python /bin/bash
```

- Compile `nms`
    
    ```bash
    cd /data/faster_rcnn/nms
    python setup.py build_ext --inplace 
    rm -rf build
    ```

- Compile `utils`
    
    ```bash
    cd /data/faster_rcnn/utils
    python setup.py build_ext --inplace 
    ```

- Compile `roi_pooling`
    
    ```bash
    cd /data/faster_rcnn/roi_pooling/src/cuda/  
    nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61

    cd /data/faster_rcnn/roi_pooling
    python setup.py build_ext --inplace 
    ```
    

    
