#!/bin/bash
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


source /setup.sh


# TODO: Make these arguments piped through from jsonnet
export MACHINE_TYPE=n1-standard-16
export ACCELERATOR_TYPE=v3-32
export RUNTIME_VERSION=pytorch-nightly
export RESOURCE_SUFFIX=$POD_UID
source /setup-pytorch-pods.sh

set -u
set -x

source /publish.sh

# "$@" should look like python -m torch_xla.distributed.xla_dist --tpu=...
docker-entrypoint.sh "$@"

# Teardown resources
/teardown-pytorch-pods.sh
