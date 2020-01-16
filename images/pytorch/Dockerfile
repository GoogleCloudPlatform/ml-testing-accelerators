FROM gcr.io/tpu-pytorch/xla:nightly

RUN curl -sSL https://sdk.cloud.google.com | bash

COPY images/common.sh /
COPY images/pytorch/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]