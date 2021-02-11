FROM python:3.6-slim

RUN apt-get update && apt-get install -y curl build-essential g++

ADD tf_*.whl .
RUN pip install *.whl
RUN rm *.whl

RUN mkdir tensorflow
WORKDIR tensorflow
RUN curl -sSL https://github.com/tensorflow/models/archive/master.tar.gz | tar xvz
RUN mv models-*/ models/

RUN pip install -r /tensorflow/models/official/requirements.txt

ENV PYTHONPATH=/tensorflow/models
WORKDIR /tensorflow/models
ENTRYPOINT ["bash"]
