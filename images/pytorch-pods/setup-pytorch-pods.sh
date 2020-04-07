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

set -u
set -e
set -x

NUM_CORES=$(echo "${ACCELERATOR_TYPE}" | egrep -o "[0-9]*" | tail -1)
NUM_CORES_PER_HOST=8
INSTANCE_GROUP_SIZE=$(expr "${NUM_CORES}" / "${NUM_CORES_PER_HOST}")

echo "Creating GCE Instance Template: ${INSTANCE_TEMPLATE_NAME}"
# Make sure it has PD with the dataset
gcloud compute --project="${PROJECT}" \
  instance-templates \
  create \
  "${INSTANCE_TEMPLATE_NAME}" \
  --machine-type="${MACHINE_TYPE}" \
  --maintenance-policy=MIGRATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --image-family=torch-xla \
  --image-project=ml-images \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name="${INSTANCE_TEMPLATE_NAME}" \
  --reservation-affinity=any \
  --metadata=enable-oslogin=TRUE

echo "Creating GCE Instance Group: ${INSTANCE_GROUP_NAME}"
gcloud compute --project="${PROJECT}" \
  instance-groups \
  managed \
  create \
  "${INSTANCE_GROUP_NAME}" \
  --base-instance-name="${INSTANCE_GROUP_NAME}" \
  --template="${INSTANCE_TEMPLATE_NAME}" \
  --size="${INSTANCE_GROUP_SIZE}" \
  --zone="${ZONE}"

echo "Waiting for ${INSTANCE_GROUP_NAME} to start..."
PROJECT=${PROJECT} INSTANCE_GROUP_SIZE=${INSTANCE_GROUP_SIZE} ZONE=${ZONE} \
  timeout 120 \
  bash -c 'while [[ ${size} != ${INSTANCE_GROUP_SIZE} ]]; \
    do sleep 10 && \
    size=$(gcloud compute instance-groups \
      list-instances \
      ${INSTANCE_GROUP_NAME} \
      --zone=${ZONE} \
      --filter="STATUS=RUNNING" \
      --format="value(NAME)" \
      | wc -l) && \
    echo "$size/${INSTANCE_GROUP_SIZE} instances started..."; done'

# GKE will wait until the TPU is READY, but not necessarily until it is HEALTHY
echo "Waiting for TPU Pod ${TPU_POD_NAME} to become healthy..."
PROJECT=${PROJECT} TPU_POD_NAME=${TPU_POD_NAME} ZONE=${ZONE} \
  timeout 120 \
  bash -c 'while [[ ${health:-NONE} != "HEALTHY" ]]; \
    do sleep 10 && \
    health=$(gcloud \
      --project=${PROJECT} \
      compute \
      tpus \
      describe \
      ${TPU_POD_NAME} \
      --zone=${ZONE} \
      --format="value(health)") && \
    echo "Waiting for healthy TPU (current health ${health:-NONE})..."; done'
