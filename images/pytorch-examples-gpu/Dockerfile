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

FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4

RUN apt-get update && apt-get install -y --no-install-recommends git
RUN curl -sSL https://sdk.cloud.google.com | bash

# Checkout examples repo at HEAD.
RUN git clone https://github.com/pytorch/examples.git
RUN pip3 install torch torchvision
RUN pip3 install -r /examples/imagenet/requirements.txt

COPY images/setup.sh /
COPY images/pytorch-examples-gpu/entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
