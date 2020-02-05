#!/bin/bash

export PUBSUB_MESSAGE="{
  \"model_dir\": \"$MODEL_DIR\",
  \"test_name\": \"$TEST_NAME\",
  \"logs_link\": \"$STACKDRIVER_LOGS\",
  \"job_name\": \"$JOB_NAME\",
  \"metric_collection_config\": $METRIC_COLLECTION_CONFIG,
  \"regression_test_config\": $REGRESSION_TEST_CONFIG
}"
/root/google-cloud-sdk/bin/gcloud pubsub topics publish metrics-written --message "$PUBSUB_MESSAGE"
