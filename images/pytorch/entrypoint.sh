#!/bin/bash
set -e

source /common.sh

# Trim grpc:// prefix
export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"

set -u
set -x

docker-entrypoint.sh "$@"

export PUBSUB_MESSAGE="{
  \"model_dir\": \"$MODEL_DIR\",
  \"test_name\": \"$TEST_NAME\",
  \"logs_link\": \"$STACKDRIVER_LOGS\",
  \"job_name\": \"$JOB_NAME\",
  \"metric_collection_config\": $METRIC_COLLECTION_CONFIG,
  \"regression_test_config\": $REGRESSION_TEST_CONFIG
}"
/root/google-cloud-sdk/bin/gcloud pubsub topics publish metrics-written --message "$PUBSUB_MESSAGE"
