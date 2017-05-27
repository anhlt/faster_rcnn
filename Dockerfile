FROM continuumio/anaconda
RUN apt-get -y update && apt-get install -y g++ gcc gfortran build-essential git libopenblas-dev
RUN conda install python -y
RUN conda install seaborn -y
RUN conda install pyqt=4.10.4 -y
RUN conda install pytorch torchvision -c soumith -y
RUN conda install opencv -y

WORKDIR /tmp/
RUN git clone https://github.com/pdollar/coco
WORKDIR coco/PythonAPI
RUN python setup.py install

VOLUME ["/data"]

ADD ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp/
RUN pip install -r requirements.txt
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git --user 
