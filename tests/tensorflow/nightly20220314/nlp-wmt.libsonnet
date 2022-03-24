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
  local transformer = common.TfNlpTest {
    modelName: 'nlp-wmt-transformer',
    scriptConfig+: {
      experiment: 'wmt_transformer/large',
      paramsOverride+: {
        task+: {
          sentencepiece_model_path: '$(TRANSFORMER_DIR)/ende_bpe_32k.model',
        },
      },
    },
  },
  local functional = common.Functional {
    scriptConfig+: {
      paramsOverride+: {
        trainer+: {
          train_steps: 10000,
          validation_interval: 10000,
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
  configs: [
    //transformer + accelerator + functional
    //for accelerator in [v2_8, v3_8, v2_32, v3_32]
  ] + [
    //transformer + v2_8 + convergence,
    //transformer + v3_8 + convergence,
    //transformer + v2_32 + convergence,
    //transformer + v3_32 + convergence,
  ],
}
