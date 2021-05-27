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

local common = import 'common.libsonnet';
local gpus = import 'templates/gpus.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local experimental = import 'tests/experimental.libsonnet';

{
  local mnist = common.ModelGardenTest {
    modelName: 'mnist',
    command: [
      'python3',
      'official/vision/image_classification/mnist_main.py',
      '--data_dir=%s' % self.flags.dataDir,
      '--model_dir=%s' % self.flags.modelDir,
    ],
    flags:: {
      dataDir: '$(MNIST_DIR)',
      modelDir: '$(MODEL_DIR)',
    },
  },
  local functional = common.Functional {
    command+: [
      '--train_epochs=1',
      '--epochs_between_evals=1',
    ],
  },
  local convergence = common.Convergence {
    command+: [
      '--train_epochs=10',
      '--epochs_between_evals=10',
    ],
  },
  local v100 = {
    accelerator: gpus.teslaV100,
    command+: [
      '--num_gpus=1',
    ],
  },
  local k80 = {
    accelerator: gpus.teslaK80,
    command+: [
      '--num_gpus=1',
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      '--distribution_strategy=tpu',
      '--batch_size=1024',
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      '--distribution_strategy=tpu',
      '--batch_size=2048',
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    commmand+: [
      '--distribution_strategy=tpu',
      '--batch_size=4096',
    ],
  },

  local tpuVm = experimental.TensorFlowTpuVmMixin {
    command+: [
      '--download',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
    ],
    flags+:: {
      dataDir: '/tmp/mnist',
      modelDir: '$(LOCAL_OUTPUT_DIR)',
    },
  },

  configs: [
    mnist + k80 + functional + mixins.Experimental,
    mnist + v100 + functional,
    mnist + v2_8 + functional,
    mnist + v2_8 + convergence,
    mnist + v2_8 + convergence + tpuVm,
    mnist + v3_8 + functional,
    mnist + v3_8 + convergence,
    mnist + v2_32 + convergence + tpuVm,
  ],
}
