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

local base = import "base.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local mixins = import "templates/mixins.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local keras_test = base.GardenTest {
    testFeature:: error "Must override `testFeature`",
    modelName: "keras-api",
    command: utils.scriptCommand(
      |||
        export PATH=$PATH:/root/google-cloud-sdk/bin
        gcloud source repos clone tf2-api-tests --project=xl-ml-test
        cd tf2-api-tests
        pip3 install behave
        behave -e ipynb_checkpoints --tags=-fails -i %s
      ||| % self.testFeature
    ),
    regressionTestConfig: {
      alert_for_failed_jobs: false,
    },
  },

  local API = {
    mode: "api",
    timeout: timeouts.one_hour,
    # Run at 2AM PST daily
    schedule: "0 10 * * *",
    accelerator+: {
      preemptible: true,
    },
  },

  local connection = API {
    mode: "connection",
    testFeature:: "aaa_connection",
  },

  local custom_layers = API {
    mode: "custom-layers",
    testFeature:: "custom_layers_model"
  },

  local custom_training_loop = API {
    mode: "custom-training-loop",
    testFeature:: "custom_training_loop",
  },

  local feature_column = API {
    mode: "feature-column",
    testFeature:: "feature_column",
  },

  local rnn = API {
    mode: "rnn",
    testFeature:: "rnn",
  },

  local save_and_load = API {
    mode: "save-and-load",
    testFeature:: "save_and_load",
    # Run at 2:30AM PST daily
    schedule: "0 10 30 * *",
  },

  local train_and_evaluate = API {
    mode: "train-and-evaluate",
    testFeature:: "train_and_evaluate",
    # Run at 2:30AM PST daily
    schedule: "0 10 30 * *",
  },

  local train_validation_dataset = API {
    mode: "train-eval-dataset",
    testFeature:: "train_validation_dataset",
    # Run at 2:30AM PST daily
    schedule: "0 10 30 * *",
  },

  local transfer_learning = API {
    mode: "transfer-learning",
    testFeature:: "transfer_learning",
    # Run at 2:30AM PST daily
    schedule: "0 10 30 * *",
  },

  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },

  configs: [
    keras_test + v2_8 + connection,
    keras_test + v2_8 + custom_layers,
    keras_test + v2_8 + custom_training_loop,
    keras_test + v2_8 + feature_column,
    keras_test + v2_8 + rnn,
    keras_test + v2_8 + save_and_load + mixins.Experimental,
    keras_test + v2_8 + train_and_evaluate + mixins.Experimental,
    keras_test + v2_8 + train_validation_dataset,
    keras_test + v2_8 + transfer_learning,
    keras_test + v3_8 + connection,
    keras_test + v3_8 + custom_layers,
    keras_test + v3_8 + custom_training_loop,
    keras_test + v3_8 + feature_column,
    keras_test + v3_8 + rnn,
    keras_test + v3_8 + save_and_load + mixins.Experimental,
    keras_test + v3_8 + train_and_evaluate,
    keras_test + v3_8 + train_validation_dataset + mixins.Experimental,
    keras_test + v3_8 + transfer_learning,
  ],
}

