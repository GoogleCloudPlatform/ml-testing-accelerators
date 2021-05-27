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
local tpus = import 'templates/tpus.libsonnet';
local experimental = import '../experimental.libsonnet';

{
  local ncf = common.ModelGardenTest {
    modelName: 'ncf',
    command: [
      'python3',
      'official/recommendation/ncf_keras_main.py',
      '--train_dataset_path=$(NCF_DIR)/tpu_data_dir/training_cycle_*/*',
      '--eval_dataset_path=$(NCF_DIR)/tpu_data_dir/eval_data/*',
      '--input_meta_data_path=$(NCF_DIR)/tpu_data_dir/metadata',
      '--data_dir=$(NCF_DIR)/movielens_data',
      '--batch_size=99000',
      '--learning_rate=3e-5',
      '--dataset=ml-20m',
      '--eval_batch_size=160000',
      '--learning_rate=0.00382059',
      '--beta1=0.783529',
      '--beta2=0.909003',
      '--epsilon=1.45439e-07',
      '--num_factors=64',
      '--hr_threshold=0.635',
      '--layers=256,256,128,64',
      '--use_synthetic_data=false',
      '--model_dir=$(MODEL_DIR)',
    ],
  },
  local functional = common.Functional {
    command+: [
      '--train_epochs=1',
      '--ml_perf=true',
    ],
  },
  local convergence = common.Convergence {
    command+: [
      '--train_epochs=14',
      '--ml_perf=false',
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        'eval/hit_rate_final': {
          success_threshold: {
            fixed_value: 0.62,
          },
          comparison: 'greater',
        },
      },
    },
  },

  local ctl = {
    command+: [
      '--keras_use_ctl=true',
    ],
  },

  local no_ctl = {
    command+: [
      '--keras_use_ctl=false',
    ],
  },

  local k80 = {
    accelerator: gpus.teslaK80,
  },

  local k80x8 = {
    accelerator: gpus.teslaK80 { count: 8 },
  },
  local v100 = {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = {
    accelerator: gpus.teslaV100 { count: 4 },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  local tpuVm = experimental.TensorFlowTpuVmMixin,

  configs: [
    ncf + k80 + functional + ctl,
    ncf + k80 + convergence + no_ctl,
    ncf + k80x8 + functional + no_ctl,
    ncf + k80x8 + convergence + ctl,
    ncf + v100 + functional + ctl,
    ncf + v100 + convergence + no_ctl,
    ncf + v100x4 + functional + no_ctl,
    ncf + v100x4 + convergence + ctl,
    ncf + v2_8 + functional + ctl,
    ncf + v3_8 + convergence + ctl,
    ncf + v2_32 + convergence + ctl + tpus.reserved + { schedule: '20 15 * * 1,3,5,6' },
    ncf + v3_32 + functional + ctl,
    ncf + v2_8 + functional + ctl + tpuVm,
    ncf + v2_32 + functional + ctl + tpuVm,
  ],
}
