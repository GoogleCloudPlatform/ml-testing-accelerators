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
  local bert = common.TfNlpTest {
    modelName: 'nlp-bert-mnli',
    scriptConfig+: {
      experiment: 'bert/sentence_prediction_text',
      configFiles: [
        'official/nlp/configs/experiments/glue_mnli_text.yaml',
      ],
      paramsOverride+: {
        task+: {
          init_checkpoint+: '$(TF_NLP_BERT_DIR)/uncased_L-12_H-768_A-12/bert_model.ckpt',
          train_data+: {
            vocab_file: '$(TF_NLP_BERT_DIR)/uncased_L-12_H-768_A-12/vocab.txt',
          },
          validation_data+: {
            vocab_file: '$(TF_NLP_BERT_DIR)/uncased_L-12_H-768_A-12/vocab.txt',
          },
        },
      },
    },
  },
  local functional = common.Functional {
    scriptConfig+: {
      paramsOverride+: {
        trainer+: {
          train_steps: 2000,
          validation_interval: 1000,
        },
      },
    },
  },
  local convergence = common.Convergence,
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
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  local v4_32 = {
    accelerator: tpus.v4_32,
  },
  local tpuVm = common.tpuVm,
  configs: [
    bert + accelerator + functional
    for accelerator in [v2_8, v3_8, v2_32, v3_32]
  ] + [
    bert + v2_8 + convergence + timeouts.Hours(4),
    bert + v3_8 + convergence + timeouts.Hours(3),
    bert + v2_32 + convergence,
    bert + v3_32 + convergence,
    bert + v4_32 + convergence + tpuVm,
    bert + v4_8 + convergence + tpuVm,
    bert + v4_8 + functional + tpuVm,
    bert + v4_32 + functional + tpuVm,

  ],
}
