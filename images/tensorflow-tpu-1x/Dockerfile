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

FROM tensorflow/tensorflow:1.15.2

RUN apt-get update && apt-get install -y --no-install-recommends curl git
RUN pip install \
  pyyaml \
  google-cloud-storage \
  google-api-python-client \
  oauth2client

# Checkout tensorflow/tpu models at HEAD
WORKDIR /
RUN curl -L https://github.com/tensorflow/tpu/archive/r1.15.tar.gz | tar zx
RUN mv tpu-* tpu/
RUN pip3 install Cython matplotlib opencv-python-headless pyyaml Pillow
RUN pip3 install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
ENV PYTHONPATH /tpu/models:/tpu/models/official/efficientnet

COPY images/setup.sh /
COPY images/tensorflow/entrypoint.sh /

WORKDIR /tpu
ENTRYPOINT ["/entrypoint.sh"]
