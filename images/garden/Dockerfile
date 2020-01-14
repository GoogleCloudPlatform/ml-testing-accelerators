FROM python:3.7-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl

RUN pip3 install \
    tf-nightly-cpu \
    cloud-tpu-client \
    tensorflow-hub \
    tensorflow-model-optimization \
    tensorflow-datasets \
    google-api-python-client \
    oauth2client
RUN curl -sSL https://sdk.cloud.google.com | bash

# Checkout tensorflow/models at HEAD
RUN git clone --depth 1 --branch master https://github.com/tensorflow/models.git /garden
WORKDIR /garden
ENV PYTHONPATH /garden

COPY images/common.sh /
COPY images/garden/entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
