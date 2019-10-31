FROM pavanpkulkarni/spark_image:2.2.1
LABEL authors="pavanpkulkarni@pavanpkulkarni.com"

COPY video.py /opt/video.py
COPY enhance.py /opt/enhance.py
COPY video.mp4 /opt/video.mp4
COPY requirement.txt /opt/requirement.txt

##RUN add-apt-repository -y ppa:fkrull/deadsnakes
RUN apt-get -y update && \
    apt-get -y install libpython-dev && \
    apt-get -y install python3.5 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.5 get-pip.py

RUN apt-add-repository ppa:mc3man/trusty-media -y && \
    apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get install ffmpeg -y

#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && \
#    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip3 install -r /opt/requirement.txt
RUN pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

#COPY Docker_WordCount_Spark-1.0.jar /opt/Docker_WordCount_Spark-1.0.jar
#COPY sample.txt /opt/sample.txt
