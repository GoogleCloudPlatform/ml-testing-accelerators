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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local imagenet = self.imagenet,
  imagenet:: {
    scriptConfig+: {
      trainFilePattern: '$(IMAGENET_DIR)/train*',
      evalFilePattern: '$(IMAGENET_DIR)/valid*',
    },
  },
  local resnet = self.resnet,
  resnet:: common.TfVisionTest + imagenet {
    modelName: 'vision-resnet',
    scriptConfig+: {
      experiment: 'resnet_imagenet',
    },
  },
  local resnet_rs = self.resnet_rs,
  resnet_rs:: common.TfVisionTest + imagenet {
    modelName: 'vision-resnetrs',
    scriptConfig+: {
      experiment: 'resnet_rs_imagenet',
      configFiles: ['official/vision/configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml'],
    },
  },
  local functional = self.functional,
  functional:: common.Functional {
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
  tpuVm:: experimental.TensorFlowTpuVmMixin,

  local functionalTests = [
    benchmark + accelerator + functional
    for benchmark in [resnet, resnet_rs]
    for accelerator in [v2_8, v3_8]
  ],
  local convergenceTests = [
    resnet + v2_32 + convergence + tpuVm,
    resnet + v3_32 + convergence + tpuVm,
    resnet_rs + v2_32 + convergence + tpuVm + timeouts.Hours(15),
    resnet_rs + v3_32 + convergence + tpuVm + timeouts.Hours(15),
  ],
  configs: functionalTests + convergenceTests + [
    resnet + v4_8 + functional + tpuVm,
    resnet + v4_32 + convergence + tpuVm,
    resnet_rs + v4lite_4 + functional + tpuVm,
  ],
}
