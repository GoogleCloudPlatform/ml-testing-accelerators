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

{
  local dlrm = common.ModelGardenTest {
    modelName: 'ranking-dlrm',
    paramsOverride:: {
      runtime: {
        distribution_strategy: error 'Must set `runtime.distribution_strategy`',
      },
      task: {
        train_data: {
          input_path: '$(CRITEO_DATA_DIR)/train/*',
          global_batch_size: 16384,
        },
        validation_data: {
          input_path: '$(CRITEO_DATA_DIR)/eval/*',
          global_batch_size: 16384,
        },
        model: {
          num_dense_features: 13,
          bottom_mlp: [512, 256, 64],
          embedding_dim: 64,
          top_mlp: [1024, 1024, 512, 256, 1],
          interaction: 'dot',
          vocab_sizes: [
            39884406,
            39043,
            17289,
            7420,
            20263,
            3,
            7120,
            1543,
            63,
            38532951,
            2953546,
            403346,
            10,
            2208,
            11938,
            155,
            4,
            976,
            14,
            39979771,
            25641295,
            39664984,
            585935,
            12972,
            108,
            36,
          ],
        },
      },
      trainer: {
        use_orbit: true,
        validation_interval: 90000,
        checkpoint_interval: 270000,
        validation_steps: 5440,
        train_steps: 256054,
        optimizer_config: {
          embedding_optimizer: 'SGD',
          lr_config: {
            decay_exp: 1.6,
            decay_start_steps: 150000,
            decay_steps: 136054,
            learning_rate: 30,
            warmup_steps: 8000,
          },
        },
      },

    },
    command: [
      'python3',
      '/usr/share/models/official/recommendation/ranking/train.py',
      '--params_override=%s' % (std.manifestYamlDoc(self.paramsOverride) + '\n'),
      '--model_dir=$(MODEL_DIR)',
    ],
  },
  local functional = common.Functional {
    command+: [
      '--mode=train',
    ],
    paramsOverride+: {
      trainer+: {
        train_steps: 10000,
      },
    },
  },
  local convergence = common.Convergence {
    local config = self,

    command+: [
      '--mode=train_and_eval',
    ],
    paramsOverride+: {
      trainer+: {
        train_steps: 256054,
      },
    },
  },

  local gpu_common = {
    local config = self,

    paramsOverride+:: {
      runtime+: {
        distribution_strategy: 'mirrored',
        num_gpus: config.accelerator.count,
      },
      task+: {
        model+: {
          bottom_mlp: [512, 256, 8],
          embedding_dim: 8,
        },
      },
    },
  },

  local cross_interaction = {
    modelName: 'ranking-dcn',
    local config = self,

    paramsOverride+:: {
      task+: {
        model+: {
          interaction: 'cross',
        },
      },
    },
  },

  local v100 = gpu_common {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
  },

  local tpu_common = {
    paramsOverride+:: {
      runtime+: {
        distribution_strategy: 'tpu',
        tpu: '$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      },
    },
  },

  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
    paramsOverride+:: {
      task+: {
        model+: {
          bottom_mlp: [512, 256, 16],
          embedding_dim: 16,
        },
      },
    },
  },

  local v2_32 = tpu_common {
    accelerator: tpus.v2_32,
    paramsOverride+:: {
      task+: {
        model+: {
          bottom_mlp: [512, 256, 64],
          embedding_dim: 64,
        },
      },
    },
  },

  local v3_8 = tpu_common {
    accelerator: tpus.v3_8,
    paramsOverride+:: {
      task+: {
        model+: {
          bottom_mlp: [512, 256, 32],
          embedding_dim: 32,
        },
      },
    },
  },

  local v3_32 = tpu_common {
    accelerator: tpus.v3_32,
    paramsOverride+:: {
      task+: {
        model+: {
          bottom_mlp: [512, 256, 128],
          embedding_dim: 128,
        },
      },
    },
  },

  local v4_8 = tpu_common {
    accelerator: tpus.v4_8,
    paramsOverride+:: {
      task+: {
        model+: {
          bottom_mlp: [512, 256, 64],
          embedding_dim: 64,
        },
      },
    },
  },
  local v4_32 = tpu_common {
    accelerator: tpus.v4_32,
    paramsOverride+:: {
      task+: {
        model+: {
          bottom_mlp: [512, 256, 128],
          embedding_dim: 128,
        },
      },
    },
  },
  local tpuVm = common.tpuVm,

  configs: [
    dlrm + functional + v3_8,
    dlrm + functional + cross_interaction + v2_8,
    dlrm + convergence + v3_8,
    dlrm + convergence + cross_interaction + v3_8,
    dlrm + functional + v3_32,
    dlrm + functional + cross_interaction + v2_32,
    dlrm + convergence + cross_interaction + v3_32,
    dlrm + convergence + v3_32,
    dlrm + functional + v100,
    dlrm + functional + v100x4,
    dlrm + convergence + v100,
    dlrm + convergence + cross_interaction + v100,
    dlrm + convergence + v100x4,
    dlrm + functional + v2_8 + tpuVm,
    dlrm + functional + cross_interaction + v2_8 + tpuVm,
    dlrm + convergence + v3_8 + tpuVm,
    dlrm + convergence + v2_32 + tpuVm,
    dlrm + functional + v3_32 + tpuVm,
    dlrm + functional + v4_8 + tpuVm,
    dlrm + functional + cross_interaction + v4_32 + tpuVm,
  ],
}
