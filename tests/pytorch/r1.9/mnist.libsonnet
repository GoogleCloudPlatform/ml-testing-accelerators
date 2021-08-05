// Copyright 2021 Google LLC
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
  local mnist = common.PyTorchTest {
    modelName: 'mnist',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_mnist.py',
      '--logdir=$(MODEL_DIR)',
      '--datadir=/datasets/mnist-data',
    ],
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
  local tpuVm = common.TpuVmMixin_1_9 {
    frameworkPrefix: 'pt-r1.9',
    command: utils.scriptCommand(
      |||
        %(command_common)s
        python3 xla/test/test_train_mp_mnist.py --logdir='' --datadir=/datasets/mnist-data
      ||| % common.tpu_vm_1_9_install
    ),
  },
  local tpuVmPod = experimental.PyTorch1_9TpuVmPodTest {
    frameworkPrefix: 'pt-r1.9',
    command: utils.scriptCommand(
      |||
        sudo ls -l /datasets
        sudo ls -l /datasets/mnist-data
        python3 -m torch_xla.distributed.xla_dist --tpu=$(cat ~/tpu_name) -- python3 /usr/share/pytorch/xla/test/test_train_mp_mnist.py --logdir='' --fake_data
      |||
    ),
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    schedule: '0 23 * * *',
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    schedule: '2 23 * * *',
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    schedule: '12 17 * * *',
  },
  configs: [
    mnist + convergence + v2_8 + timeouts.Hours(1),
    mnist + convergence + v3_8 + timeouts.Hours(1),
    mnist + convergence + v2_8 + timeouts.Hours(1) + tpuVm,
    mnist + convergence + v3_8 + timeouts.Hours(1) + tpuVm,
    mnist + convergence + v3_32 + timeouts.Hours(1) + tpuVmPod,
  ],
}
