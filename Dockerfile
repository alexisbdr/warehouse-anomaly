FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y vim

RUN git clone https://github.com/alexisbdr/warehouse-anomaly && cd warehouse-anomaly

RUN pip install pillow matplotlib gdown