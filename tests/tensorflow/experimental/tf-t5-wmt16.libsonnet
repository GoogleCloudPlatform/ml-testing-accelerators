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
  local t5 = common.HuggingFaceTransformer {
    modelName: 't5-wmt16',
    command: utils.scriptCommand(
      |||
        %(initialSetup)s
        cd /tmp/transformers/examples/tensorflow/translation
        pip install -r requirements.txt
        mkdir /tmp/t5-translation
        python3 run_translation.py \
          --model_name_or_path t5-small \
          --do_train \
          --do_eval \
          --max_train_samples 10000 \
          --max_eval_samples 1000 \
          --num_train_epochs 1 \
          --num_beams 1 \
          --source_lang en \
          --target_lang ro \
          --source_prefix "translate English to Romanian: " \
          --dataset_name wmt16 \
          --dataset_config_name ro-en \
          --output_dir /tmp/t5-translation \
          --per_device_train_batch_size=64 \
          --per_device_eval_batch_size=64
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
    t5 + accelerator + common.Functional + common.tpuVm
    for accelerator in [v2_8, v3_8, v4_8]
  ],
}
