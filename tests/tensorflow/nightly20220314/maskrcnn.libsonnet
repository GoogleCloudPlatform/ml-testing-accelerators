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
  local maskrcnn = common.ModelGardenTest {
    modelName: 'maskrcnn',
    paramsOverride:: {
      eval: {
        eval_file_pattern: '$(COCO_DIR)/val*',
        val_json_file: '$(COCO_DIR)/instances_val2017.json',
      },
      train: {
        iterations_per_loop: 5000,
        checkpoint: {
          path: '$(RESNET_PRETRAIN_DIR)/resnet50-checkpoint-2018-02-07',
          prefix: 'resnet50/',
        },
        frozen_variable_prefix: '(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\\/',
        total_steps: error 'Must set `train.total_steps`',
        batch_size: error 'Must set `train.batch_size`',
        train_file_pattern: '$(COCO_DIR)/train*',
      },
      postprocess: {
        pre_nms_num_boxes: 1000,
      },
    },
    command: [
      'python3',
      'official/legacy/detection/main.py',
      '--model=mask_rcnn',
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

    paramsOverride+:: {
      architecture+: {
        use_bfloat16: false,
      },
    },
    command+: [
      '--num_gpus=%d' % config.accelerator.count,
    ],
  },
  local k80 = gpu_common {
    local config = self,

    paramsOverride+:: {
      train+: {
        batch_size: 2 * config.accelerator.replicas,
      },
      eval+: {
        batch_size: 2 * config.accelerator.replicas,
      },
      predict+: {
        batch_size: 2 * config.accelerator.replicas,
      },
    },
    accelerator: gpus.teslaK80,
  },
  local k80x8 = k80 {
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
      eval+: {
        batch_size: 4 * config.accelerator.replicas,
      },
      predict+: {
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
      eval+: {
        batch_size: 40,
      },
      predict+: {
        batch_size: 40,
      },
      architecture+: {
        use_bfloat16: true,
      },
    },
    command+: [
      '--strategy_type=tpu',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
    ],
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train+: {
        batch_size: 32,
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
        batch_size: 128,
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
  local v4_8 = tpu_common {
    accelerator: tpus.v4_8,
    paramsOverride+: {
      train+: {
        batch_size: 16,
      },
    },

  },
  local v4_32 = tpu_common {
    accelerator: tpus.v4_32,
    paramsOverride+: {
      train+: {
        batch_size: 64,
      },
    },

  },
  local tpuVm = common.tpuVm,

  configs: [
    //maskrcnn + functional + v2_8,
    //maskrcnn + functional + v3_8,
    //maskrcnn + convergence + v2_8,
    //maskrcnn + convergence + v3_8,
    //maskrcnn + functional + v2_32 + common.RunNightly,
    //maskrcnn + functional + v3_32,
    //maskrcnn + convergence + v3_32,
    maskrcnn + functional + v2_8 + tpuVm + { paramsOverride+: { train+: { batch_size: 16 } } },
    maskrcnn + functional + v2_32 + tpuVm + { paramsOverride+: { train+: { batch_size: 64 } } },
    //maskrcnn + functional + v4_8 + tpuVm,
    //maskrcnn + functional + v4_32 + tpuVm,
    //maskrcnn + convergence + v4_8 + tpuVm,
    //maskrcnn + convergence + v4_32 + tpuVm,
  ],
}
