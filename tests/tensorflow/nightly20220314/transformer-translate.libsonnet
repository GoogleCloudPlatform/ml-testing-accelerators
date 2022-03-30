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
  local transformer = common.ModelGardenTest {
    modelName: 'transformer-translate',
    command: [
      'python3',
      'official/nlp/transformer/transformer_main.py',
      '--param_set=big',
      '--max_length=64',
      '--data_dir=$(TRANSFORMER_DIR)',
      '--vocab_file=$(TRANSFORMER_DIR)/vocab.ende.32768',
      '--bleu_source=$(TRANSFORMER_DIR)/newstest2014.en',
      '--bleu_ref=$(TRANSFORMER_DIR)/newstest2014.de',
      '--enable_tensorboard',
      '--model_dir=$(MODEL_DIR)',
    ],
  },
  local functional = common.Functional {
    command+: [
      '--train_steps=20000',
    ],
  },
  local functional_short = common.Functional {
    command+: [
      '--train_steps=10000',
    ],
  },
  local convergence = common.Convergence {
    local config = self,

    command+: [
      '--train_steps=%d' % (200000 / config.accelerator.replicas),
    ],
  },

  local gpu_common = {
    local config = self,

    command+: [
      '--num_gpus=%d' % config.accelerator.count,
      '--steps_between_evals=5000',
    ],
  },
  local k80 = gpu_common {
    local config = self,

    accelerator: gpus.teslaK80,
    command+: [
      '--batch_size=%d' % (2048 * config.accelerator.replicas),
    ],
  },
  local k80x8 = k80 {
    accelerator: gpus.teslaK80 { count: 8 },
    command+: [
      '--all_reduce_alg=hierarchical_copy',
    ],
  },
  local v100 = gpu_common {
    local config = self,

    accelerator: gpus.teslaV100,
    command+: [
      '--batch_size=%d' % (4096 * config.accelerator.replicas),
    ],
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
  },

  local tpu_common = {
    command+: [
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--distribution_strategy=tpu',
      '--steps_between_evals=10000',
      '--static_batch=true',
      '--use_ctl=true',
      '--padded_decode=true',
      '--decode_batch_size=32',
      '--decode_max_length=97',
    ],
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
    command+: [
      '--batch_size=6144',
    ],
  },
  local v3_8 = tpu_common {
    accelerator: tpus.v3_8,
    command+: [
      '--batch_size=6144',
    ],
  },
  local v2_32 = tpu_common {
    accelerator: tpus.v2_32,
    command+: [
      '--batch_size=24576',
    ],
  },
  local v3_32 = tpu_common {
    accelerator: tpus.v3_32,
    command+: [
      '--batch_size=24576',
    ],
  },
  local tpuVm = common.tpuVm,

  configs: [
    //transformer + v2_8 + functional,
    transformer + v2_8 + functional + tpuVm,
    transformer + v3_8 + functional + tpuVm,
    //transformer + v2_8 + convergence,
    //transformer + v3_8 + convergence,
    //transformer + v2_32 + functional + common.RunNightly,
    transformer + v2_32 + functional + tpuVm,
    //transformer + v3_32 + functional,
    //transformer + v3_32 + convergence,
  ],
}
