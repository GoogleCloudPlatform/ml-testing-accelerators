// Copyright 2022 Google LLC
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
  local coco = {
    scriptConfig+: {
      trainFilePattern: '$(COCO_DIR)/train*',
      evalFilePattern: '$(COCO_DIR)/val*',
    },
  },
  local tpu_common = {
    local config = self,
    scriptConfig+: {
      paramsOverride+: {
        task+: {
          validation_data+: {
            global_batch_size: 8 * config.accelerator.replicas,
          },
        },
      },
    },
  },
  local retinanet = common.TfVisionTest + coco {
    modelName: 'vision-retinanet',
    scriptConfig+: {
      experiment: 'retinanet_resnetfpn_coco',
      paramsOverride+: {
        task+: {
          annotation_file: '$(COCO_DIR)/instances_val2017.json',
        },
      },
    },
  },
  local maskrcnn = common.TfVisionTest + coco {
    modelName: 'vision-maskrcnn',
    scriptConfig+: {
      experiment: 'maskrcnn_resnetfpn_coco',
      paramsOverride+: {
        task+: {
          annotation_file: '$(COCO_DIR)/instances_val2017.json',
        },
      },
    },
  },
  local functional = common.Functional {
    scriptConfig+: {
      paramsOverride+: {
        trainer+: {
          train_steps: 200,
          validation_interval: 200,
          validation_steps: 100,
        },
      },
    },
  },
  local convergence = common.Convergence,
  local v2_8 = {
    accelerator: tpus.v2_8,
    scriptConfig+: {
      paramsOverride+: {
        task+: {
          train_data+: {
            global_batch_size: 64,
          },
        },
      },
    },
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
  local v4_8 = tpu_common {
    accelerator: tpus.v4_8,
  },
  local v4_32 = tpu_common {
    accelerator: tpus.v4_32,
  },
  local tpuVm = experimental.TensorFlowTpuVmMixin,

  local functionalTests = [
    benchmark + accelerator + functional
    for benchmark in [retinanet, maskrcnn]
    for accelerator in [v2_8, v3_8, v2_32, v3_32]
  ],
  local convergenceTests = [
    retinanet + v2_8 + convergence + timeouts.Hours(24),
    retinanet + v3_8 + convergence + timeouts.Hours(24),
    retinanet + v2_32 + convergence + timeouts.Hours(15),
    retinanet + v3_32 + convergence + timeouts.Hours(15),
    maskrcnn + v2_8 + convergence + timeouts.Hours(24),
    maskrcnn + v3_8 + convergence + timeouts.Hours(24),
    maskrcnn + v2_32 + convergence + timeouts.Hours(15),
    maskrcnn + v3_32 + convergence + timeouts.Hours(15),
  ],
  configs: functionalTests + convergenceTests + [
    retinanet + v4_8 + functional + tpuVm,
    retinanet + v4_32 + functional + tpuVm,
    maskrcnn + v4_8 + functional + tpuVm,
    maskrcnn + v4_32 + functional + tpuVm,
  ],
}
