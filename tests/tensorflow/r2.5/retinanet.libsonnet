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
  local retinanet = common.ModelGardenTest {
    modelName: 'retinanet',
    paramsOverride:: {
      local params = self,

      eval: {
        eval_file_pattern: '$(COCO_DIR)/val*',
        batch_size: params.train.batch_size,
        val_json_file: '$(COCO_DIR)/instances_val2017.json',
      },
      predict: {
        batch_size: params.train.batch_size,
      },
      train: {
        checkpoint: {
          path: '$(RESNET_PRETRAIN_DIR)/resnet50-checkpoint-2018-02-07',
          prefix: 'resnet50/',
        },
        total_steps: error 'Must set `train.total_steps`',
        batch_size: error 'Must set `train.batch_size`',
        train_file_pattern: '$(COCO_DIR)/train*',
      },
    },
    command: [
      'python3',
      'official/vision/detection/main.py',
      '--params_override=%s' % (std.manifestYamlDoc(self.paramsOverride) + '\n'),
      '--model_dir=$(MODEL_DIR)',
    ],
  },
  local functional = common.Functional {
    command+: [
      '--mode=train',
    ],
    paramsOverride+: {
      train+: {
        total_steps: 1000,
      },
    },
  },
  local convergence = common.Convergence {
    local config = self,

    command+: [
      '--mode=train',
    ],
    paramsOverride+: {
      train+: {
        total_steps: 22500 / config.accelerator.replicas,
      },
    },
  },

  local gpu_common = {
    local config = self,

    command+: [
      '--num_gpus=%d' % config.accelerator.count,
    ],
  },
  local k80x8 = gpu_common {
    local config = self,

    paramsOverride+:: {
      train+: {
        batch_size: 4 * config.accelerator.replicas,
      },
    },
    accelerator: gpus.teslaK80 { count: 8 },
    command+: [
      '--all_reduce_alg=hierarchical_copy',
    ],
  },
  local v100 = gpu_common {
    local config = self,

    paramsOverride+:: {
      train+: {
        batch_size: 4 * config.accelerator.replicas,
      },
    },
    accelerator: gpus.teslaV100,
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
  },

  local tpu_common = {
    paramsOverride+:: {
      architecture+: {
        use_bfloat16: true,
      },
    },
    command+: [
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--strategy_type=tpu',
    ],
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train+: {
        batch_size: 64,
      },
    },
  },
  local v3_8 = tpu_common {
    accelerator: tpus.v3_8,
    paramsOverride+: {
      train+: {
        batch_size: 64,
      },
    },
  },
  local v2_32 = tpu_common {
    accelerator: tpus.v2_32,
    paramsOverride+: {
      train+: {
        batch_size: 256,
      },
    },
  },
  local v3_32 = tpu_common {
    accelerator: tpus.v3_32,
    paramsOverride+: {
      train+: {
        batch_size: 256,
      },
    },
  },

  configs: [
    retinanet + functional + k80x8 + mixins.Suspended,
    retinanet + convergence + k80x8 + mixins.Experimental,
    retinanet + functional + v100 + mixins.Suspended,
    retinanet + functional + v100x4 + mixins.Suspended,
    retinanet + convergence + v100x4 + mixins.Experimental,
    retinanet + functional + v2_8,
    retinanet + functional + v3_8,
    retinanet + convergence + v2_8,
    retinanet + convergence + v3_8,
    retinanet + functional + v2_32,
    retinanet + functional + v3_32,
    retinanet + convergence + v3_32,
  ],
}
