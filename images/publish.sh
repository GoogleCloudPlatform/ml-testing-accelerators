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


export PUBSUB_MESSAGE="{
  \"model_dir\": \"$MODEL_DIR\",
  \"test_name\": \"$TEST_NAME\",
  \"logs_link\": \"$STACKDRIVER_LOGS\",
  \"job_name\": \"$JOB_NAME\",
  \"job_namespace\": $POD_NAMESPACE,
  \"metric_collection_config\": $METRIC_COLLECTION_CONFIG,
  \"regression_test_config\": $REGRESSION_TEST_CONFIG
}"
/root/google-cloud-sdk/bin/gcloud pubsub topics publish metrics-written --message "$PUBSUB_MESSAGE"
