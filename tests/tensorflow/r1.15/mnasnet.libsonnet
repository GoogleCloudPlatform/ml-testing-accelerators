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
local tpus = import 'templates/tpus.libsonnet';

{
  local mnasnet = common.LegacyTpuTest {
    modelName: 'mnasnet',
    command: [
      'python3',
      '/tpu/models/official/mnasnet/mnasnet_main.py',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--iterations_per_loop=1000',
      '--mode=train',
      '--data_dir=$(IMAGENET_DIR)',
      '--model_dir=$(MODEL_DIR)',
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      '--config_file=/tpu/models/official/mnasnet/configs/cloud/v2-8.yaml',
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      '--config_file=/tpu/models/official/mnasnet/configs/cloud/v3-8.yaml',
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      '--config_file=/tpu/models/official/mnasnet/configs/cloud/v2-32.yaml',
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      '--config_file=/tpu/models/official/mnasnet/configs/cloud/v3-32.yaml',
    ],
  },
  local convergence = common.Convergence,
  local functional = common.Functional {
    command+: [
      '--train_steps=1000',
    ],
  },

  configs: [
    mnasnet + v2_8 + convergence,
    mnasnet + v3_8 + convergence,
    mnasnet + v2_32 + convergence,
    mnasnet + v3_32 + convergence,
    mnasnet + v2_8 + functional,
    mnasnet + v3_8 + functional,
    mnasnet + v2_32 + functional,
    mnasnet + v3_32 + functional,
  ],
}
