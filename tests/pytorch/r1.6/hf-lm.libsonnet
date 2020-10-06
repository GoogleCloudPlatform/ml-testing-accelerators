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
local tpus = import "templates/tpus.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local command_common = |||
    git clone -b v3.2.0 https://github.com/huggingface/transformers.git
    cd transformers && pip install .
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
  local bert_large_fp32 = common.Convergence {
    modelName: "hf-bert-large-fp32",
    command: utils.scriptCommand(
      |||
        %(common)s --mlm --model_type=bert \
        --model_name_or_path bert-large-uncased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
  },
  local bert_large_bf16 = common.Convergence {
    modelName: "hf-bert-large-bf16",
    command: utils.scriptCommand(
      |||
        %(common)s --mlm --model_type=bert \
        --model_name_or_path bert-large-uncased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              envMap+: {
                XLA_USE_BF16: "1",
              },
            },
          },
        },
      },
    },
  },
  local gpt2_medium_fp32 = common.Convergence {
    modelName: "hf-gpt2-medium-fp32",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=gpt \
        --model_name_or_path gpt2-medium \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
  },
  local gpt2_medium_bf16 = common.Convergence {
    modelName: "hf-gpt2-medium-bf16",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=gpt2 \
        --model_name_or_path gpt2-medium \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              envMap+: {
                XLA_USE_BF16: "1",
              },
            },
          },
        },
      },
    },
  },

  local xlnet_large_fp32 = common.Convergence {
    modelName: "hf-xlnet-large-fp32",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=xlnet \
        --model_name_or_path xlnet-large-cased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --model_max_length 512
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
  },
  local xlnet_large_bf16 = common.Convergence {
    modelName: "hf-xlnet-large-bf16",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=xlnet \
        --model_name_or_path xlnet-large-cased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --model_max_length 512
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              envMap+: {
                XLA_USE_BF16: "1",
              },
            },
          },
        },
      },
    },
  },

  local hf_lm = common.PyTorchTest {
    modelName: "hf-lm",
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              resources+: {
                requests: {
                  cpu: "48.0",
                  memory: "100Gi",
                },
              },
            },
          },
        },
      },
    },
  },

  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    hf_lm + v3_8 + bert_large_fp32 + timeouts.Hours(24),
    hf_lm + v3_8 + bert_large_bf16 + timeouts.Hours(24),
    hf_lm + v3_8 + gpt2_medium_fp32 + timeouts.Hours(24),
    hf_lm + v3_8 + gpt2_medium_bf16 + timeouts.Hours(24),
  ],
}
