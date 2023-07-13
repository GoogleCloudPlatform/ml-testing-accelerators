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

ARG image_version="nightly"

FROM tensorflow/tensorflow:${image_version}-gpu

ARG image_version
ARG model_garden_branch="master"
ARG tensorflow_text_version="nightly"

# TODO (ericlefort) remove this once the base image gets updated to 22.04
# Keep track of the full set of dependencies in 3.8 and reinstall them on 3.10
RUN pip3 freeze > requirements.txt
RUN apt-get install software-properties-common && apt-get update && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends git python3.10 python3.10-distutils python3.10-dev libcairo2-dev
RUN cd /usr/bin &&  unlink python3 && ln -s /usr/bin/python3.10 python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN cat requirements.txt | xargs -n 1 pip3 install
RUN rm requirements.txt

RUN pip3 install cloud-tpu-client pyyaml

# HACKs: Remove this once they are added to the Model Garden requirements
RUN pip3 install tensorflow-recommenders --no-deps
RUN if [ ${image_version} = "nightly" ] \
    || [ ${tensorflow_text_version} = "nightly" ] \
    || [ echo ${tensorflow_text_version} | grep -q "dev" ]; then \
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

