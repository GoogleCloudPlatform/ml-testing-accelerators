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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local resnetrs = common.ModelGardenTest {
    modelName: 'classifier-resnetrs',
    paramsOverride:: {
      trainer: {
        train_steps: error 'Must set `trainer.train_steps`',
        validation_interval: error 'Must set `trainer.validation_interval`',
      },
      task: {
        train_data: {
          input_path: '$(IMAGENET_DIR)/train*',
        },
        validation_data: {
          input_path: '$(IMAGENET_DIR)/valid*',
        },
      },
    },
    command: [
      'python3',
      'official/vision/beta/train.py',
      '--experiment=resnet_rs_imagenet',
      '--mode=train_and_eval',
      '--model_dir=$(MODEL_DIR)',
      '--config_file=official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml',
      '--params_override=%s' % std.manifestYamlDoc(self.paramsOverride) + '\n',
    ],
  },
  local functional = common.Functional {
    paramsOverride+: {
      trainer+: {
        train_steps: 320,
        validation_interval: 320,
      },
    },
  },
  local convergence = common.Convergence {
    paramsOverride+: {
      trainer+: {
        train_steps: 109200,
        validation_interval: 3120,
      },
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'validation/accuracy': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.79,
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

  local gpu_common = {
    local config = self,

    paramsOverride+:: {
      runtime+: {
        distribution_strategy: 'mirrored',
        loss_scale: 'dynamic',
        num_gpus: config.accelerator.count,
      },
    },
  },
  local k80 = gpu_common {
    local config = self,

    paramsOverride+:: {
      task+: {
        train_data+: {
          global_batch_size: 128,
        },
        validation_data+: {
          global_batch_size: 128,
        },
      },
    },
    accelerator: gpus.teslaK80,
  },
  local k80x8 = k80 {
    paramsOverride+:: {
      runtime+: {
        all_reduce_alg: 'hierarchical_copy',
      },
    },
    accelerator: gpus.teslaK80 { count: 8 },
  },
  local v100 = gpu_common {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
  },
  local v100x8 = v100 {
    accelerator: gpus.teslaV100 { count: 8 },
  },

  local tpu_common = {
    command+: [
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
    ],
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
  },
  local v3_8 = tpu_common {
    accelerator: tpus.v3_8,
  },
  local v2_32 = tpu_common {
    accelerator: tpus.v2_32,
  },
  local v3_32 = tpu_common {
    accelerator: tpus.v3_32,
  },

  configs: [
    resnetrs + v2_8 + functional + mixins.Unsuspended,
    resnetrs + v3_8 + functional,
    resnetrs + v2_8 + convergence + timeouts.Hours(30),
    resnetrs + v3_8 + convergence + timeouts.Hours(30),
    resnetrs + v2_32 + functional,
    resnetrs + v3_32 + functional,
    resnetrs + v3_32 + convergence + timeouts.Hours(15),
  ],
}
