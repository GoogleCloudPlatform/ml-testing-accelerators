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

ARG image_version=nightly
FROM tensorflow/tensorflow:${image_version}-gpu

ARG model_garden_branch=master

RUN apt-get update && apt-get install -y --no-install-recommends git

RUN pip3 install cloud-tpu-client pyyaml

ARG tensorflow_text_version=nightly

# HACKs: Remove this once they are added to the Model Garden requirements
RUN pip3 install tensorflow-recommenders --no-deps
RUN if [[ "${image_version}" == "nightly" || "${tensorflow_text_version}" == "nightly" || "${tensorflow_text_version}" == *"dev"* ]]; then \
    pip3 install tensorflow-text-nightly --no-deps; \
    else \
    pip3 install tensorflow-text==${tensorflow_text_version} --no-deps; \
    fi

# Checkout tensorflow/models at HEAD
WORKDIR /
RUN curl -L https://github.com/tensorflow/models/archive/${model_garden_branch}.tar.gz | tar zx
RUN mv models-${model_garden_branch} garden/

RUN pip3 install -r /garden/official/requirements.txt
ENV PYTHONPATH /garden

RUN curl -L https://github.com/tensorflow/benchmarks/archive/master.tar.gz | tar xz
RUN mv benchmarks-master benchmarks/

COPY images/setup.sh /
COPY images/tensorflow/entrypoint.sh /

RUN curl -L https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-289.0.0-linux-x86_64.tar.gz | tar zx
ENV PATH "/google-cloud-sdk/bin/:${PATH}"

WORKDIR /garden
ENTRYPOINT ["/entrypoint.sh"]
