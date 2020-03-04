# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends git jq
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

COPY images/setup.sh /
COPY images/publish.sh /
COPY images/garden/entrypoint.sh /

WORKDIR /garden
ENTRYPOINT ["/entrypoint.sh"]
