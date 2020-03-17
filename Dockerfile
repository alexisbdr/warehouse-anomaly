FROM tensorflow/tensorflow:latest-gpu

RUN apt install git

RUN git clone https://github.com/alexisbdr/warehouse-anomaly && cd warehouse-anomaly

RUN pip install -r requirements.txt