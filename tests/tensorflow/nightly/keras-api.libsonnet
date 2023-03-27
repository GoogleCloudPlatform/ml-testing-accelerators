// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local keras_test = self.keras_test,
  keras_test:: common.ModelGardenTest {
    testFeature:: error 'Must override `testFeature`',
    modelName: 'keras-api',
    isTPUPod:: error 'Must set `isTPUPod`',
    command: utils.scriptCommand(
      |||
        cd ~
        export PATH=$PATH:/root/google-cloud-sdk/bin
        export PATH=$PATH:/home/xl-ml-test/.local/bin
        export TPU_NAME=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)
        gcloud source repos clone tf2-api-tests --project=xl-ml-test
        cd tf2-api-tests
        pip3 install behave
        behave -e ipynb_checkpoints --tags=-fails %s -i %s
      ||| % [if self.isTPUPod then '--tags=-failspod' else '', self.testFeature]
    ),
  },

  local API = self.API,
  API:: common.RunNightly {
    mode: 'api',
    timeout: timeouts.one_hour,
    tpuSettings+: {
      preemptible: true,
    },
  },

  local connection = self.connection,
  connection:: API {
    mode: 'connection',
    testFeature:: 'aaa_connection',
  },

  local custom_layers = self.custom_layers,
  custom_layers:: API {
    mode: 'custom-layers',
    testFeature:: 'custom_layers_model',
  },

  local custom_training_loop = self.custom_training_loop,
  custom_training_loop:: API {
    mode: 'ctl',
    testFeature:: 'custom_training_loop',
  },

  local feature_column = self.feature_column,
  feature_column:: API {
    mode: 'feature-column',
    testFeature:: 'feature_column',
  },

  local rnn = self.rnn,
  rnn:: API {
    mode: 'rnn',
    testFeature:: 'rnn',
  },

  local preprocessing_layers = self.preprocessing_layers,
  preprocessing_layers:: API {
    mode: 'preprocess-layers',
    testFeature:: 'preprocessing_layers',
  },

  local upsample = self.upsample,
  upsample:: API {
    mode: 'upsample',
    testFeature:: 'upsample',
  },

  local save_load_io_device_local = self.save_load_io_device_local,
  save_load_io_device_local:: API {
    mode: 'save-load-localhost',
    testFeature:: 'save_and_load_io_device_local_drive',
  },

  local save_and_load = self.save_and_load,
  save_and_load:: API {
    mode: 'save-and-load',
    testFeature:: 'save_and_load.feature',
  },

  local train_and_evaluate = self.train_and_evaluate,
  train_and_evaluate:: API {
    mode: 'train-and-evaluate',
    testFeature:: 'train_and_evaluate',
  },

  local train_validation_dataset = self.train_validation_dataset,
  train_validation_dataset:: API {
    mode: 'train-eval-dataset',
    testFeature:: 'train_validation_dataset',
  },

  local transfer_learning = self.transfer_learning,
  transfer_learning:: API {
    mode: 'transfer-learning',
    testFeature:: 'transfer_learning',
  },

  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
    isTPUPod: false,
  },

  local v2_32 = self.v2_32,
  v2_32:: {
    accelerator: tpus.v2_32,
    isTPUPod: true,
  },

  local tpuVm = experimental.TensorFlowTpuVmMixin,

  configs: [
    keras_test + v2_8 + connection,
    keras_test + v2_8 + connection + tpuVm,
    keras_test + v2_8 + custom_layers,
    keras_test + v2_8 + custom_layers + tpuVm,
    keras_test + v2_8 + custom_training_loop,
    keras_test + v2_8 + custom_training_loop + tpuVm,
    keras_test + v2_8 + feature_column + timeouts.Hours(2),
    keras_test + v2_8 + feature_column + timeouts.Hours(2) + tpuVm,
    keras_test + v2_8 + preprocessing_layers,
    keras_test + v2_8 + upsample,
    keras_test + v2_8 + upsample + tpuVm,
    keras_test + v2_8 + rnn,
    keras_test + v2_8 + rnn + tpuVm,
    keras_test + v2_8 + save_and_load + timeouts.Hours(2),
    keras_test + v2_8 + save_and_load + timeouts.Hours(2) + tpuVm,
    keras_test + v2_8 + save_load_io_device_local + timeouts.Hours(2),
    keras_test + v2_8 + save_load_io_device_local + timeouts.Hours(2) + tpuVm,
    keras_test + v2_8 + train_and_evaluate + timeouts.Hours(3),
    keras_test + v2_8 + train_and_evaluate + timeouts.Hours(3) + tpuVm,
    keras_test + v2_8 + train_validation_dataset,
    keras_test + v2_8 + transfer_learning,
    keras_test + v2_8 + transfer_learning + tpuVm,
  ],
}
