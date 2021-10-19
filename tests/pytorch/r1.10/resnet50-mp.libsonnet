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
local gpus = import 'templates/gpus.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local gpu_command_base = |||
    unset XRT_TPU_CONFIG
    export GPU_NUM_DEVICES=%(num_gpus)s
    python3 pytorch/xla/test/test_train_mp_imagenet.py \
    --model=resnet50 \
    --batch_size=128 \
    --log_steps=100 \
    --num_workers=2 \
    --num_epochs=2 \
    --datadir=/datasets/imagenet-mini \
  |||,
  local resnet50_gpu_py37_cuda_101 = common.PyTorchTest {
    imageTag: 'r1.10_3.7_cuda_10.1',
    modelName: 'resnet50-mp-cuda-10-1',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '7.0',
    memory: '40Gi',

  },
  local resnet50_gpu_py37_cuda_102 = common.PyTorchTest {
    imageTag: 'r1.10_3.7_cuda_10.2',
    modelName: 'resnet50-mp-cuda-10-2',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '7.0',
    memory: '40Gi',

  },
  local resnet50_gpu_py37_cuda_112 = common.PyTorchTest {
    imageTag: 'r1.10_3.7_cuda_11.2',
    modelName: 'resnet50-mp-cuda-11-2',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '7.0',
    memory: '40Gi',

  },
  local resnet50_MP = common.PyTorchTest {
    modelName: 'resnet50-mp',
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_imagenet.py',
      '--logdir=$(MODEL_DIR)',
      '--model=resnet50',
      '--num_workers=8',
      '--batch_size=128',
      '--log_steps=200',
    ],
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: '90.0',
                memory: '400Gi',
              },
            },
          },
        },
      },
    },
  },

  local functional = common.Functional {
    command+: [
      '--num_epochs=2',
      '--datadir=/datasets/imagenet-mini',
    ],
  },
  local convergence = common.Convergence {
    command+: [
      '--num_epochs=90',
      '--datadir=/datasets/imagenet',
    ],
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Accuracy/test': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 75.0,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
  local resnet50_tpu_vm = common.PyTorchTest {
    frameworkPrefix: 'pt-r1.10',
    modelName: 'resnet50-mp',
    paramsOverride: {
      num_epochs: error 'Must set `num_epochs`',
      datadir: error 'Must set `datadir`',
      setup_commands: error 'Must set `setup_commands`',
    },
    command: utils.scriptCommand(
      |||
        %(setup_commands)s
        pip3 install tensorboardX google-cloud-storage
        python3 xla/test/test_train_mp_imagenet.py \
          --logdir=$(MODEL_DIR) \
          --datadir=%(datadir)s \
          --model=resnet50 \
          --num_workers=8 \
          --batch_size=128 \
          --log_steps=200 \
          --num_epochs=%(num_epochs)d \
      ||| % self.paramsOverride,
    ),
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: '1',
                memory: '2Gi',
              },
            },
          },
        },
      },
    },
  },
  local functional_tpu_vm = common.Functional {
    paramsOverride: {
      setup_commands: common.tpu_vm_1_10_install,
      num_epochs: 2,
      datadir: '/datasets/imagenet-mini',
    },
  },
  local convergence_tpu_vm = common.Convergence {
    paramsOverride: {
      setup_commands: common.tpu_vm_1_10_install,
      num_epochs: 5,
      datadir: '/datasets/imagenet',
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Accuracy/test': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 30.0,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
            aten_ops_sum: {
              FINAL: {
                wait_for_n_data_points: 0,
                inclusive_bounds: true,
                fixed_value: {
                  comparison: 'LESS',
                  value: 40.0,
                },
              },
            },
          },
        },
      },
    },

  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  local v100 = {
    accelerator: gpus.teslaV100,
    command: utils.scriptCommand(
      gpu_command_base % 1
    ),
  },
  local v100_amp = {
    accelerator: gpus.teslaV100,
    command: utils.scriptCommand(
      |||
        %(gpu_command_base)s --amp
      ||| % (gpu_command_base % 1)
    ),
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
    command: utils.scriptCommand(
      gpu_command_base % 4
    ),
  },
  configs: [
    resnet50_MP + v3_8 + convergence + timeouts.Hours(26) + mixins.PreemptibleTpu,
    resnet50_MP + v3_8 + functional + timeouts.Hours(2),
    resnet50_gpu_py37_cuda_101 + common.Functional + v100 + timeouts.Hours(2),
    resnet50_gpu_py37_cuda_101 + common.Functional + v100_amp + timeouts.Hours(2) + { modelName: 'resnet50-cuda-10-1-amp' },
    resnet50_gpu_py37_cuda_101 + common.Functional + v100x4 + timeouts.Hours(1),
    resnet50_gpu_py37_cuda_102 + common.Functional + v100 + timeouts.Hours(2),
    resnet50_gpu_py37_cuda_102 + common.Functional + v100_amp + timeouts.Hours(2) + { modelName: 'resnet50-cuda-10-2-amp' },
    resnet50_gpu_py37_cuda_102 + common.Functional + v100x4 + timeouts.Hours(1),
    resnet50_gpu_py37_cuda_112 + common.Functional + v100 + timeouts.Hours(2),
    resnet50_gpu_py37_cuda_112 + common.Functional + v100x4 + timeouts.Hours(1),
    resnet50_tpu_vm + v3_8 + functional_tpu_vm + timeouts.Hours(2) + experimental.PyTorchTpuVmMixin,
    resnet50_tpu_vm + v3_8 + convergence_tpu_vm + timeouts.Hours(4) + experimental.PyTorchTpuVmMixin,
  ],
}
