FROM python:3.7-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  gcc \
  git \
  python3.7-dev \
  libglib2.0-0
RUN curl -sSL https://sdk.cloud.google.com | bash

RUN pip3 install tf-nightly-cpu cloud-tpu-client pyyaml

# Checkout tensorflow/models at HEAD
RUN git clone --depth 1 --branch master https://github.com/tensorflow/models.git /garden
WORKDIR /garden
RUN pip3 install -r official/requirements.txt
ENV PYTHONPATH /garden

COPY images/common.sh /
COPY images/garden/entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
