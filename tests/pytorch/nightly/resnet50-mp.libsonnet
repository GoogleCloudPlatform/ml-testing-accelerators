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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local resnet50 = common.PyTorchTest {
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

    cpu: '90.0',
    memory: '400Gi',
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

  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },

  local gpu = {
    local config = self,
    imageTag+: '_cuda_11.2',

    cpu: '7.0',
    memory: '40Gi',

    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              GPU_NUM_DEVICES: '%d' % config.accelerator.count,
            },
          },
        },
      },
    },
  },
  local v100x4 = gpu {
    accelerator: gpus.teslaV100 { count: 4 },
  },

  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip install tensorboardX google-cloud-storage
      |||,
    },
  },

  configs: [
    resnet50 + functional + v100x4 + timeouts.Hours(1),
    resnet50 + functional + v3_8 + timeouts.Hours(2) + tpuVm,
    resnet50 + convergence + v3_8 + timeouts.Hours(24) + tpuVm,
    resnet50 + functional + v3_32 + timeouts.Hours(1) + tpuVm,
    resnet50 + convergence + v3_32 + timeouts.Hours(12) + tpuVm,
  ],
}
