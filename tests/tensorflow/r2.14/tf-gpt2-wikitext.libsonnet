// Copyright 2023 Google LLC
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
  local gpt2 = common.HuggingFaceTransformer {
    modelName: 'gpt2-wikitext',
    command: utils.scriptCommand(
      |||
        %(initialSetup)s
        cd /tmp/transformers/examples/tensorflow/language-modeling
        pip install -r requirements.txt
        mkdir /tmp/gpt2-wikitext
        python3 run_clm.py \
          --model_name_or_path distilgpt2 \
          --max_train_samples 1000 \
          --max_eval_samples 100 \
          --num_train_epochs 1 \
          --output_dir /tmp/gpt2-wikitext \
          --dataset_name wikitext \
          --dataset_config_name wikitext-103-raw-v1
      ||| % self.script,
    ),
  },

  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    gpt2 + accelerator + common.Functional + common.tpuVm
    for accelerator in [v2_8, v3_8, v4_8]
  ],
}
