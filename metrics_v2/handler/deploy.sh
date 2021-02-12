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

staging_dir=$(mktemp -d)

set -x
tar xf handler/gcf_bundle.tar -C "${staging_dir}"
gcloud functions deploy wcromar-test-function --entry-point=receive_test_event --source "${staging_dir}" --runtime python38 --trigger-topic wcromar-test-topic --set-env-vars BQ_DATASET=wcromar_test_dataset --project=xl-ml-test --memory=4096MB
