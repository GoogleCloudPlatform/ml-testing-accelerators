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

local common = import '../common.libsonnet';
local latest_common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
{
  local hf_bert_mrpc = latest_common.hfBertCommon {
    modelName+: '.mrpc',
    extraFlags+: ['--task_name mrpc', '--max_seq_length 128', '--eval_steps 100'],
  },

  local functional = mixins.Functional {
    extraFlags+: ['--num_train_epochs 1'],
    extraConfig:: 'default.py',
  },

  local convergence = mixins.Convergence {
    extraConfig:: 'default.py',
    extraFlags+: ['--num_train_epochs 3', '--learning_rate 2e-5', '--eval_steps 500'],
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/accurary': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.84,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },

  local v2 = common.tpuVmBaseImage {
    extraFlags+:: ['--per_device_train_batch_size 4', '--per_device_eval_batch_size 4'],
  },
  local v3 = common.tpuVmBaseImage {
    extraFlags+:: ['--per_device_train_batch_size 4', '--per_device_eval_batch_size 4'],
  },
  local v4 = common.tpuVmV4Base {
    extraFlags+:: ['--per_device_train_batch_size 8', '--per_device_eval_batch_size 8'],
  },

  local v2_8 = v2 {
    accelerator: tpus.v2_8,
  },
  local v3_8 = v3 {
    accelerator: tpus.v3_8,
  },
  local v4_8 = v4 {
    accelerator: tpus.v4_8,
  },
  local v4_32 = v4 {
    accelerator: tpus.v4_32,
  },

  configs: [
    hf_bert_mrpc + convergence + v4_32,
    hf_bert_mrpc + functional + v4_8,
    hf_bert_mrpc + functional + v3_8,
    hf_bert_mrpc + functional + v2_8,
  ],
}
