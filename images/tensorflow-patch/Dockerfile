# Copyright 2021 Google LLC
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

FROM python:3.6

RUN apt update && apt install -y python3-dev

ARG tensorflow_minor_version=r2.4

RUN git clone https://github.com/tensorflow/tensorflow.git --branch=$tensorflow_minor_version --depth=1
RUN bash -c "cd tensorflow && source tensorflow/tools/ci_build/release/common.sh && install_ubuntu_16_pip_deps"
RUN curl -sSLO https://github.com/bazelbuild/bazel/releases/download/$(cat /tensorflow/.bazelversion)/bazel-$(cat /tensorflow/.bazelversion)-installer-linux-x86_64.sh
RUN chmod +x bazel-*.sh
RUN ./bazel-*.sh

WORKDIR /tensorflow

RUN ./configure

# Using more than 16 jobs on Cloud Build causes OOM.
ARG bazel_build_jobs=16
RUN bazel build --config=opt --jobs=$bazel_build_jobs //tensorflow/tools/pip_package:build_pip_package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN ls /tmp/tensorflow_pkg

RUN pip install /tmp/tensorflow_pkg/*.whl

# Model Garden does not create new branches for patch releases.
WORKDIR /
RUN curl -L https://github.com/tensorflow/models/archive/$tensorflow_minor_version.0.tar.gz | tar zx
RUN mv models-$tensorflow_minor_version.0 models/
RUN pip3 install -r /models/official/requirements.txt
ENV PYTHONPATH /models
WORKDIR /models

RUN rm -rf /tensorflow /bazel-*.sh

COPY images/tensorflow/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
