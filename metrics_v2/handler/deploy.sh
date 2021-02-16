#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -u

function usage {
  echo "Usage: deploy.sh --name <FUNCTION_NAME> --project <GCP_PROJECT> --topic <PUBSUB_TOPIC> --dataset <BQ_DATASET>"
  exit 1
}

while test $# -gt 0; do
  case "$1" in
    --name)
      name=$2
      shift 2
      ;;
    --project)
      project=$2
      shift 2
      ;;
    --topic)
      topic=$2
      shift 2
      ;;
    --dataset)
      dataset=$2
      shift 2
      ;;
    *)
      echo "Invalid flag: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z $name || -z $project || -z $topic || -z $dataset ]]; then
  echo "Required flag not provided."
  usage
fi

echo "Function Name: ${name}"
echo "GCP Project: ${project}"
echo "PubSub Topic: ${topic}"
echo "BQ Dataset: ${dataset}"

staging_dir=$(mktemp -d)

set -x
tar xf handler/gcf_bundle.tar -C "${staging_dir}"
gcloud functions deploy "${name}" --project="${project}" --source "${staging_dir}" --entry-point=receive_test_event --runtime python38 --trigger-topic "${topic}" --set-env-vars "BQ_DATASET=${dataset}" --memory=4096MB 
