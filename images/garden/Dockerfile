FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends git
RUN curl -sSL https://sdk.cloud.google.com | bash

RUN pip3 install cloud-tpu-client pyyaml

# Checkout tensorflow/models at HEAD
WORKDIR /
RUN curl -LO https://github.com/tensorflow/models/archive/master.zip
RUN unzip master.zip
RUN rm master.zip
RUN mv models-master garden/
RUN pip3 install -r /garden/official/requirements.txt
ENV PYTHONPATH /garden

COPY images/common.sh /
COPY images/garden/entrypoint.sh /

WORKDIR /garden
ENTRYPOINT ["/entrypoint.sh"]
