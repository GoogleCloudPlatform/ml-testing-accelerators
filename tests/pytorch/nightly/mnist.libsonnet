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

local common = import "common.libsonnet";
local experimental = import "tests/experimental.libsonnet";
local gpus = import "templates/gpus.libsonnet";
local mixins = import "templates/mixins.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local mnist = common.PyTorchTest {
    modelName: 'mnist',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    command: [
      "python3",
      "pytorch/xla/test/test_train_mp_mnist.py",
      "--logdir=%s" % self.flags.modelDir,
      "%s" % self.flags.dataset,
    ],
    flags:: {
      modelDir: "$(MODEL_DIR)",
      dataset: "--datadir=/datasets/mnist-data"
    },
  },

  local gpu_command_base = |||
    unset XRT_TPU_CONFIG
    export GPU_NUM_DEVICES=%(num_gpus)s
    python3 pytorch/xla/test/test_train_mp_mnist.py \
      --datadir=/datasets/mnist-data
  |||,

  local mnist_gpu = common.PyTorchTest {
    imageTag: 'nightly_3.6_cuda',
    modelName: 'mnist',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
  },

  local convergence = common.Convergence {
    regressionTestConfig+: {
      metric_success_conditions+: {
        'Accuracy/test_final': {
          success_threshold: {
            fixed_value: 98.0,
          },
          comparison: 'greater',
        },
      },
    },
  },

  local v2_8 = {
    accelerator: tpus.v2_8,
    schedule: '10 17 * * *',
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    schedule: '4 17 * * *',
  },
  local v100 = {
    accelerator: gpus.teslaV100,
    command: utils.scriptCommand(
      gpu_command_base % 1
    ),
    schedule: '0 19 * * *',
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
    command: utils.scriptCommand(
      gpu_command_base % 4
    ),
    schedule: '2 19 * * *',
  },

  local tpuVm = experimental.PyTorchTpuVmTest {
    command: utils.scriptCommand(
      |||
        git clone https://github.com/pytorch/xla.git
        python3 xla/test/test_train_mp_mnist.py --logdir='' --datadir=/datasets/mnist-data
      |||
    ),
  },

  configs: [
    mnist + convergence + v2_8 + timeouts.Hours(1),
    mnist + convergence + v2_8 + timeouts.Hours(1) + tpuVm,
    mnist + convergence + v3_8 + timeouts.Hours(1) + tpuVm,
    mnist + convergence + v3_8 + timeouts.Hours(1),
    mnist_gpu + convergence + v100 + timeouts.Hours(1),
    mnist_gpu + convergence + v100x4 + timeouts.Hours(1),
  ],
}
