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
local utils = import 'templates/utils.libsonnet';

{
  local transformer = common.LegacyTpuTest {
    local config = self,

    modelName: 'transformer',
    command: [
      't2t-trainer',
      '--model=transformer',
      '--hparams_set=transformer_packed_tpu',
      '--problem=translate_ende_wmt32k_packed',
      '--use_tpu=True',
      '--schedule=train',
      '--data_dir=$(T2T_TRANSFORMER_DIR)',
      '--output_dir=$(MODEL_DIR)',
    ],

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+:: {
            train+: {
              args: utils.scriptCommand(
                // HACK: Trim TPU name from `zone/name` to `name`.
                |||
                  unset KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
                  %s --cloud_tpu_name="${TPU_NAME##*/}"
                ||| % std.join(' ', config.command)
              ),
            },
          },
        },
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      '--iterations_per_loop=100',
      '--tpu_num_shards=8',
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      '--iterations_per_loop=100',
      '--tpu_num_shards=8',
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      '--iterations_per_loop=5000',
      '--tpu_num_shards=32',
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      '--iterations_per_loop=5000',
      '--tpu_num_shards=32',
    ],
  },
  local convergence = common.Convergence,
  local functional = common.Functional {
    command+: [
      '--train_steps=10',
    ],
  },

  configs: [
    transformer + v2_8 + convergence + { command+: ['--train_steps=250000'] },
    transformer + v3_8 + convergence + { command+: ['--train_steps=250000'] },
    transformer + v2_32 + convergence + { command+: ['--train_steps=62500'] } + tpus.reserved + { schedule: '13 18 * * 0,2,4' },
    transformer + v3_32 + convergence + { command+: ['--train_steps=62500'] },
    transformer + v2_8 + functional,
    transformer + v3_8 + functional,
    transformer + v2_32 + functional,
    transformer + v3_32 + functional,
  ],
}
