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
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local resnet = common.TfVisionTest + common.imagenet {
    modelName: 'resnet-imagenet',
    scriptConfig+: {
      experiment: 'resnet_imagenet',
    },
  },
  local functional = common.Functional {
    scriptConfig+: {
      paramsOverride+: {
        trainer: {
          train_steps: 320,
          validation_interval: 320,
        },
      },
    },
  },
  local convergence = self.convergence,
  convergence:: common.Convergence,
  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
    scriptConfig+: {
      paramsOverride+: {
        task+: {
          train_data+: {
            global_batch_size: 1024,
          },
          validation_data+: {
            global_batch_size: 1024,
          },
        },
      },
    },
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  local v2_32 = self.v2_32,
  v2_32:: {
    accelerator: tpus.v2_32,
  },
  local v3_32 = self.v3_32,
  v3_32:: {
    accelerator: tpus.v3_32,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },
  local v4_32 = self.v4_32,
  v4_32:: {
    accelerator: tpus.v4_32,
  },
  local tpuVm = self.tpuVm,
  tpuVm:: common.tpuVm,

  local functionalTests = [
    resnet + v2_8 + functional,
    resnet + v3_8 + functional,
  ],
  local convergenceTests = [
    resnet + v2_32 + convergence + tpuVm,
    resnet + v3_32 + convergence + tpuVm,
  ],
  configs: functionalTests + convergenceTests + [
    resnet + v4_8 + functional + tpuVm,
    resnet + v4_32 + convergence + tpuVm,
  ],
}
