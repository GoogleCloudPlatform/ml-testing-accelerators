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
  local vit = common.HuggingFaceTransformer {
    modelName: 'vit-beans',
    command: utils.scriptCommand(
      |||
        %(initialSetup)s
        cd /tmp/transformers/examples/tensorflow/image-classification
        pip install -r requirements.txt
        python3 run_image_classification.py \
            --dataset_name beans \
            --output_dir ./beans_outputs/ \
            --remove_unused_columns False \
            --do_train \
            --do_eval \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --logging_strategy steps \
            --logging_steps 10 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --save_total_limit 3 
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
    vit + accelerator + common.Functional + common.tpuVm
    for accelerator in [v2_8, v3_8, v4_8]
  ],
}
