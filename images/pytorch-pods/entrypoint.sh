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

set -u
set -x

export RESOURCE_SUFFIX=$POD_UID
export INSTANCE_TEMPLATE_NAME="instance-template-${RESOURCE_SUFFIX}"
export TPU_POD_NAME="tpu-pod-${RESOURCE_SUFFIX}"
export INSTANCE_GROUP_NAME="instance-group-${RESOURCE_SUFFIX}"

# zone/name -> name
export TPU_POD_NAME=$(echo ${TPU_NAME} | awk -F'/' '{print $2}')

#export PATH=\"$PATH:/opt/google-cloud-sdk/bin\" && \
timeout 180 /setup-instances.sh && \
export master=$(gcloud compute instance-groups \
  list-instances \
  ${INSTANCE_GROUP_NAME} \
  --zone=${ZONE} \
  --format="value(NAME)" \
  --limit=1) && \
gcloud -q compute ssh --internal-ip --zone=$ZONE $master \
  --command "source /anaconda3/etc/profile.d/conda.sh && \
  conda activate ${CONDA_ENV} && \
  echo $PATH && \
  echo $PATH && \
  conda env list && \
  gcloud --version && \
  python -m torch_xla.distributed.xla_dist --tpu=${TPU_POD_NAME} --conda-env=${CONDA_ENV} ${XLA_DIST_FLAGS} -- $*"
exit_code=$?

/teardown-instances.sh
exit $exit_code
