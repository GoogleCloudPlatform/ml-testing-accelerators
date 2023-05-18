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
  local maskrcnn = common.TfVisionTest + common.coco {
    modelName: 'maskrcnn-coco',
    scriptConfig+: {
      experiment: 'maskrcnn_resnetfpn_coco',
    },
  },
  local functional = common.Functional {
    scriptConfig+: {
      paramsOverride+: {
        trainer+: {
          train_steps: 400,
          validation_interval: 200,
          validation_steps: 100,
        },
      },
    },
  },
  local convergence = common.Convergence,
  local v4_8 = tpu_common {
    accelerator: tpus.v4_8,
  },
  local v4_32 = tpu_common {
    accelerator: tpus.v4_32,
  },
  local tpuVm = common.tpuVm,

  configs: [
    maskrcnn + v4_8 + functional + tpuVm,
    maskrcnn + v4_32 + convergence + tpuVm,
  ],
}
