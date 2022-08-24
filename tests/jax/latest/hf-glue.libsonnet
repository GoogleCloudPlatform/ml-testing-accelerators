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
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';
{
  local hf_bert_common = common.JaxTest + common.huggingFace {
    local config = self,
    frameworkPrefix: 'flax-latest',
    modelName:: 'hf-bert',
    extraFlags:: '',
    testScript:: |||
      %(installPackages)s
      pip install -r examples/flax/text-classification/requirements.txt
      %(verifySetup)s

      export GCS_BUCKET=$(MODEL_DIR)
      export OUTPUT_DIR='./bert-glue'

      python3 examples/flax/text-classification/run_flax_glue.py --model_name_or_path bert-base-cased \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size 4 \
        %(extraFlags)s
      gsutil -m cp -r ${OUTPUT_DIR} $(MODEL_DIR)
    ||| % (self.scriptConfig { extraFlags: config.extraFlags }),
  },

  local functional = mixins.Functional {
    extraFlags+: '--num_train_epochs 1 ',
    extraConfig:: 'default.py',
  },

  local convergence = mixins.Convergence {
    extraConfig:: 'default.py',
    extraFlags+: '--num_train_epochs 3 --learning_rate 2e-5 --eval_steps 500 ',
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

  local mnli = {
    modelName+: '-mnli',
    extraFlags+: '--task_name mnli --max_seq_length 512 --eval_steps 1000 ',
  },
  local mrpc = {
    modelName+: '-mrpc',
    extraFlags+: '--task_name mrpc --max_seq_length 128 ',
  },

  local v2 = common.tpuVmBaseImage {
    extraFlags+:: '--per_device_train_batch_size 4 --per_device_eval_batch_size 4',
  },
  local v3 = common.tpuVmBaseImage {
    extraFlags+:: '--per_device_train_batch_size 4 --per_device_eval_batch_size 4',
  },
  local v4 = common.tpuVmV4Base {
    extraFlags+:: '--per_device_train_batch_size 8 --per_device_eval_batch_size 8',
  },

  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },

  configs: [
    hf_bert_common + mnli + convergence + v4 + v4_8 + timeouts.Hours(10),
    hf_bert_common + mrpc + convergence + v4 + v4_8 + timeouts.Hours(1),
    hf_bert_common + mnli + functional + v4 + v4_8,
    hf_bert_common + mrpc + functional + v4 + v4_8,

    hf_bert_common + mrpc + convergence + v3 + v3_8 + timeouts.Hours(2),
    hf_bert_common + mnli + functional + v3 + v3_8,
    hf_bert_common + mrpc + functional + v3 + v3_8,

    hf_bert_common + mrpc + convergence + v2 + v2_8 + timeouts.Hours(2),
    hf_bert_common + mnli + functional + v2 + v2_8,
    hf_bert_common + mrpc + functional + v2 + v2_8,
  ],
}
