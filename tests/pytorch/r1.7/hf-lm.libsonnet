# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

local common = import "common.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local command_common = |||
    git clone --recursive -b r1.7 https://github.com/pytorch-tpu/examples.git
    cd examples/deps/transformers && pip install .
    git log -1
    python examples/xla_spawn.py \
      --num_cores 8 \
      examples/language-modeling/run_language_modeling.py \
      --logging_dir ./tensorboard-metrics \
      --cache_dir ./cache_dir \
      --train_data_file /datasets/wikitext-103-raw/wiki.train.raw \
      --do_train \
      --do_eval \
      --eval_data_file /datasets/wikitext-103-raw/wiki.valid.raw \
      --overwrite_output_dir \
      --output_dir language-modeling \
      --logging_steps 100 \
      --save_steps 3000 \
      --overwrite_cache \
      --tpu_metrics_debug \
  |||,
  local command_copy_metrics = |||
    gsutil -m cp -r ./tensorboard-metrics/* $(MODEL_DIR)
  |||,
  local roberta_base_fine = common.Convergence {
    modelName: "hf-mlm-roberta-b-fine",
    command: utils.scriptCommand(
      |||
        %(common)s --mlm --model_type=roberta \
        --model_name_or_path roberta-base \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      required_metrics: ['eval_loss_final'],
      metric_success_conditions+: {
        "eval_loss_final": {
          success_threshold: {
            fixed_value: 1.5,
          },
          comparison: "less",
        },
      },
    },
  },
  local roberta_base_pre = common.Convergence {
    modelName: "hf-mlm-roberta-b-pre",
    command: utils.scriptCommand(
      |||
        %(common)s --mlm --model_type=roberta \
        --tokenizer=roberta-base \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      required_metrics: ['eval_loss_final'],
      metric_success_conditions+: {
        "eval_loss_final": {
          success_threshold: {
            fixed_value: 6.5,
          },
          comparison: "less",
        },
      },
    },
  },
  local hf_lm = common.PyTorchTest {
    modelName: "hf-lm",
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: "12.0",
    memory: "80Gi",
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    hf_lm + v3_8 + roberta_base_pre + timeouts.Hours(4),
    hf_lm + v2_8 + roberta_base_pre + timeouts.Hours(5),
    hf_lm + v3_8 + roberta_base_fine + timeouts.Hours(3),
  ],
}
