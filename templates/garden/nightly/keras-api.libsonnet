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
local timeouts = import "../../timeouts.libsonnet";
local mixins = import "../../mixins.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local base_command = ||| 
    export PATH=$PATH:/root/google-cloud-sdk/bin &&
    gcloud source repos clone cloudtpu-tf20-api-tests --project=gcp-tpupods-demo &&
    cd cloudtpu-tf20-api-tests &&
    pip3 install behave &&
  |||,

  local keras_test = base.GardenTest {
    testFeature:: "aaa",
    modelName: "keras-api",
    command: [
      "/bin/bash",
      "-c",
      base_command + " behave -e ipynb_checkpoints --tags=-fails -i %s" % self.testFeature,
    ],
  },

  local connection = mixins.Functional {
    mode: "connection",
    testFeature:: "aaa_connection",
  },

  local custom_layers = mixins.Functional {
    mode: "custom-layers",
    testFeature:: "custom_layers_model"
  },

  local custom_training_loop = mixins.Functional {
    mode: "custom-training-loop",
    testFeature:: "custom_training_loop",
  },

  local feature_column = mixins.Functional {
    mode: "feature-column",
    testFeature:: "feature_column",
  },

  local rnn = mixins.Functional {
    mode: "rnn",
    testFeature:: "rnn",
  },

  local save_and_load = mixins.Functional {
    mode: "save-and-load",
    testFeature:: "save_and_load",
  },

  local train_and_evaluate = mixins.Functional {
    mode: "train-and-evaluate",
    testFeature:: "train_and_evaluate",
  },

  local train_validation_dataset = mixins.Functional {
    mode: "train-validation-dataset",
    testFeature:: "train_validation_dataset",
  },

  local transfer_learning = mixins.Functional {
    mode: "transfer-learning",
    testFeature:: "transfer_learning",
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
    keras_test + v2_8 + save_and_load,
    keras_test + v2_8 + train_and_evaluate,
    keras_test + v2_8 + train_validation_dataset,
    keras_test + v2_8 + transfer_learning,
    keras_test + v3_8 + connection,
    keras_test + v3_8 + custom_layers,
    keras_test + v3_8 + custom_training_loop,
    keras_test + v3_8 + feature_column,
    keras_test + v3_8 + rnn,
    keras_test + v3_8 + save_and_load,
    keras_test + v3_8 + train_and_evaluate,
    keras_test + v3_8 + train_validation_dataset,
    keras_test + v3_8 + transfer_learning,
  ],
}

