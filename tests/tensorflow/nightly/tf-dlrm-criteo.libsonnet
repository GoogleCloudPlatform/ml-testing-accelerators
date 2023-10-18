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
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local dlrm = common.TfRankingTest {
    modelName: 'dlrm-criteo',
    paramsOverride+:: {
      task+: {
        model+: {
          interaction: 'dot',
        },
      },
    },
  },
  local functional = self.functional,
  functional:: common.Functional {
    command+: [
      '--mode=train',
    ],
    paramsOverride+: {
      trainer+: {
        train_steps: 10000,
      },
    },
  },
  local convergence = self.convergence,
  convergence:: common.Convergence {
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

  local tpu_common = self.tpu_common,
  tpu_common:: {
    paramsOverride+:: {
      runtime+: {
        distribution_strategy: 'tpu',
        tpu: '$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      },
    },
  },

  local v2_8 = self.v2_8,
  v2_8:: tpu_common {
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

  local v2_32 = self.v2_32,
  v2_32:: tpu_common {
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

  local v4_8 = self.v4_8,
  v4_8:: tpu_common {
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
  local v4_32 = self.v4_32,
  v4_32:: tpu_common {
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
  local tpuVm = self.tpuVm,
  tpuVm:: common.tpuVm,

  configs: [
    dlrm + functional + v2_8 + tpuVm,
    dlrm + convergence + v2_32 + tpuVm,
    dlrm + functional + v4_8 + tpuVm,
  ],
}
