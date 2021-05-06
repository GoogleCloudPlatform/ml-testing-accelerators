FROM python:3.8-slim

RUN apt-get update && apt-get install -y curl build-essential g++

ADD tf_*.whl .
RUN pip install *.whl
RUN rm *.whl

RUN mkdir tensorflow
WORKDIR tensorflow
RUN curl -sSL https://github.com/tensorflow/models/archive/master.tar.gz | tar xvz
RUN mv models-*/ models/

RUN pip install -r /tensorflow/models/official/requirements.txt
RUN apt-get install -y unzip
RUN export PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip && \
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP && \
unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
rm -f $PROTOC_ZIP

RUN apt-get install -y google-perftools
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"

ENV PYTHONPATH=/tensorflow/models
WORKDIR /tensorflow/models
ENTRYPOINT ["bash"]
