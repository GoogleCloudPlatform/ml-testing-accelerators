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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local resnet50 = common.PyTorchTest {
    modelName: 'resnet50-mp',
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_imagenet.py',
      '--model=resnet50',
      '--log_steps=200',
    ] + if self.flags.modelDir != null then [
      '--logdir=%s' % self.flags.modelDir,
    ] else [],
    flags:: {
      modelDir: '$(MODEL_DIR)',
    },
    volumeMap+: {
      datasets: common.datasetsVolume,
    },

    cpu: '90.0',
    memory: '400Gi',
  },

  local fake_data = common.Functional {
    mode: 'fake',
    command+: [
      '--fake_data',
    ],
  },
  local functional = common.Functional {
    command+: [
      '--num_epochs=2',
      '--datadir=/datasets/imagenet-mini',
    ],
  },
  local convergence = common.Convergence {
    local config = self,

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
                  // Larger global batch size gives lower final accuracy
                  value: if self.accelerator.replicas == 1 then 75 else 74,
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
  local v4_8 = {
    accelerator: tpus.v4_8,
    // Keep same global batch size as v3
    command+: ['--batch_size=256'],
  },

  local gpu = {
    local config = self,
    imageTag+: '_cuda_11.2',

    cpu: '7.0',
    memory: '40Gi',

    // Disable XLA metrics report on GPU
    command+: [
      '--nometrics_debug',
    ],
    flags+: {
      modelDir: null,
    },

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

  local nosummaries = {
    flags+:: {
      modelDir: null,
    },
  },

  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip install tensorboardX google-cloud-storage
      |||,
    },
  },
  local pjrt = tpuVm + experimental.PjRt {
    modelName: 'resnet50-pjrt',
    command: [
      'python3',
      'pytorch/xla/test/pjrt/test_train_pjrt_imagenet.py',
    ] + super.command[2:],
  },

  configs: [
    resnet50 + functional + v100x4 + timeouts.Hours(1),
    resnet50 + functional + v3_8 + timeouts.Hours(2) + tpuVm,
    resnet50 + fake_data + nosummaries + v3_8 + timeouts.Hours(2) + tpuVm,
    resnet50 + fake_data + nosummaries + v3_8 + timeouts.Hours(2) + pjrt,
    resnet50 + convergence + nosummaries + v3_8 + timeouts.Hours(24) + tpuVm,
    resnet50 + convergence + nosummaries + v3_8 + timeouts.Hours(24) + pjrt,
    resnet50 + functional + v3_32 + timeouts.Hours(1) + tpuVm,
    resnet50 + convergence + v3_32 + timeouts.Hours(12) + tpuVm,
    resnet50 + fake_data + nosummaries + v4_8 + timeouts.Hours(2) + tpuVm,
    resnet50 + fake_data + nosummaries + v4_8 + timeouts.Hours(2) + pjrt,
    resnet50 + convergence + nosummaries + v4_8 + timeouts.Hours(24) + tpuVm,
    resnet50 + convergence + nosummaries + v4_8 + timeouts.Hours(24) + pjrt,
  ],
}
