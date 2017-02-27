FROM continuumio/anaconda
RUN apt-get -y update && apt-get install -y g++ gcc gfortran build-essential git libopenblas-dev
RUN conda install python -y
RUN conda install seaborn -y
RUN conda install pyqt=4.10.4 -y
RUN conda install pytorch torchvision -c soumith -y
VOLUME ["/data"]

ADD . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git --user 
WORKDIR /data
CMD ./startjupyter.sh
