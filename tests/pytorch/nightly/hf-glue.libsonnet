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
    git clone https://github.com/huggingface/transformers.git
    cd transformers && pip install .
    git log -1
    python examples/xla_spawn.py \
      --num_cores 8 \
      examples/text-classification/run_glue.py \
      --logging_dir=./tensorboard-metrics \
      --task_name MNLI \
      --data_dir /datasets/glue/MNLI \
      --cache_dir ./cache_dir \
      --do_train \
      --do_eval \
      --num_train_epochs 3 \
      --max_seq_length 128 \
      --learning_rate 3e-5 \
      --output_dir MNLI \
      --overwrite_output_dir \
      --logging_steps 100 \
      --save_steps 3000 \
      --overwrite_cache \
      --tpu_metrics_debug \
  |||,
  local command_copy_metrics = |||
    gsutil -m cp -r ./tensorboard-metrics/* $(MODEL_DIR)
  |||,
  local bert_base_cased = common.Convergence {
    modelName: "hf-glue-bert-b-c",
    command: utils.scriptCommand(
      |||
        %(common)s --model_name_or_path bert-base-cased \
        --per_gpu_train_batch_size 64 \
        --per_gpu_eval_batch_size 64
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "eval_mnli/acc": {
          success_threshold: {
            fixed_value: 0.80,
          },
          comparison: "greater",
        },
        "eval_mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.80,
          },
          comparison: "greater",
        },
      },
    },
  },
  local xlnet_large_cased = common.Convergence {
    modelName: "hf-glue-xlnet-l-c",
    command: utils.scriptCommand(
      |||
        %(common)s --model_name_or_path xlnet-large-cased \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 32
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "eval_mnli/acc": {
          success_threshold: {
            fixed_value: 0.85,
          },
          comparison: "greater",
        },
        "eval_mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.85,
          },
          comparison: "greater",
        },
      },
    },
  },
  local roberta_large = common.Convergence {
    modelName: "hf-glue-roberta-l",
    command: utils.scriptCommand(
      |||
        %(common)s --model_name_or_path roberta-large \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 16
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "eval_mnli/acc": {
          success_threshold: {
            fixed_value: 0.85,
          },
          comparison: "greater",
        },
        "eval_mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.85,
          },
          comparison: "greater",
        },
      },
    },
  },
  local xlm_mlm_en_2048 = common.Convergence {
    modelName: "hf-glue-xlm-mlm-en-2048",
    command: utils.scriptCommand(
      |||
        %(common)s --model_name_or_path xlm-mlm-en-2048 \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 8
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "eval_mnli/acc": {
          success_threshold: {
            fixed_value: 0.80,
          },
          comparison: "greater",
        },
        "eval_mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.80,
          },
          comparison: "greater",
        },
      },
    },
  },
  local distilbert_base_uncased = common.Convergence {
    modelName: "hf-glue-distilbert-b-uc",
    command: utils.scriptCommand(
      |||
        %(common)s --model_name_or_path distilbert-base-uncased \
        --per_gpu_train_batch_size 512 \
        --per_gpu_eval_batch_size 512
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
    regressionTestConfig+: {
      required_metrics: ['eval_mnli/acc'],
      metric_success_conditions+: {
        "eval_mnli/acc": {
          success_threshold: {
            fixed_value: 0.70,
          },
          comparison: "greater",
        },
        "eval_mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.70,
          },
          comparison: "greater",
        },
      },
    },
  },
  local hf_glue = common.PyTorchTest {
    modelName: "hf-glue",
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
                  cpu: "12.0",
                  memory: "80Gi",
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
    hf_glue + v3_8 + bert_base_cased + timeouts.Hours(2),
    hf_glue + v2_8 + bert_base_cased + timeouts.Hours(2),
    hf_glue + v3_8 + xlnet_large_cased + timeouts.Hours(5),
    hf_glue + v3_8 + roberta_large + timeouts.Hours(4),
    hf_glue + v3_8 + xlm_mlm_en_2048 + timeouts.Hours(7),
    hf_glue + v3_8 + distilbert_base_uncased + timeouts.Hours(2),
  ],
}
