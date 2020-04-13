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
set -e

runtime_vars="{
  \"model_dir\": \"$MODEL_DIR\",
  \"logs_link\": \"$STACKDRIVER_LOGS\",
  \"job_name\": \"$JOB_NAME\",
  \"job_namespace\": \"$POD_NAMESPACE\",
  \"zone\": \"$ZONE\",
  \"cluster_name\": \"$CLUSTER\"
}"
pubsub_message=`echo ${runtime_vars} | jq ". + ${METRIC_CONFIG}"`
echo "Pubsub message: ${pubsub_message}"

set -x
gcloud pubsub topics publish metrics-written --message "$pubsub_message"
