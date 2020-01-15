#!/bin/bash
set -e

source /common.sh

set -u
set -x

$@

export PUBSUB_MESSAGE="{
  \"model_dir\": \"$MODEL_DIR\",
  \"config_name\": \"$CONFIG_NAME\",
  \"logs_link\": \"$STACKDRIVER_LOGS\",
  \"metric_collection_config\": $METRIC_COLLECTION_CONFIG,
  \"regression_test_config\": $REGRESSION_TEST_CONFIG
}"
/root/google-cloud-sdk/bin/gcloud pubsub topics publish metrics-written --message "$PUBSUB_MESSAGE"
