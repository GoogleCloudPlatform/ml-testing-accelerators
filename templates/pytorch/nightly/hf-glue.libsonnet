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

local base = import "base.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";
local utils = import "../../utils.libsonnet";

{
  local command_common = |||
    git clone https://github.com/huggingface/transformers.git
    cd transformers && pip install .
    git log -1
    python examples/run_tpu_glue.py \
      --tensorboard_logdir=$(MODEL_DIR) \
      --task_name MNLI \
      --do_train \
      --do_eval \
      --data_dir /datasets/glue/MNLI \
      --max_seq_length 128 \
      --learning_rate 3e-5 \
      --num_train_epochs 3.0 \
      --output_dir MNLI \
      --overwrite_output_dir \
      --logging_steps 50 \
      --save_steps 1000 \
      --num_cores=8 \
      --only_log_master \
      --metrics_debug \
  |||,
  local bert_base_cased = base.Convergence {
    modelName: "hf-glue-bert-b-c",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type bert \
        --model_name_or_path bert-base-cased \
        --train_batch_size 32 \
        --eval_batch_size 32
      ||| % {common: command_common}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "mnli/acc": {
          success_threshold: {
            fixed_value: 0.75,
          },
          comparison: "greater",
        },
        "mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.75,
          },
          comparison: "greater",
        },
      },
    },
  },
  local xlnet_large_cased = base.Convergence {
    modelName: "hf-glue-xlnet-l-c",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type xlnet \
        --model_name_or_path xlnet-large-cased \
        --train_batch_size 32 \
        --eval_batch_size 32
      ||| % {common: command_common}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "mnli/acc": {
          success_threshold: {
            fixed_value: 0.85,
          },
          comparison: "greater",
        },
        "mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.85,
          },
          comparison: "greater",
        },
      },
    },
  },
  local roberta_large = base.Convergence {
    modelName: "hf-glue-roberta-l",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type roberta \
        --model_name_or_path roberta-large \
        --train_batch_size 16 \
        --eval_batch_size 16
      ||| % {common: command_common}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "mnli/acc": {
          success_threshold: {
            fixed_value: 0.30,
          },
          comparison: "greater",
        },
        "mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.30,
          },
          comparison: "greater",
        },
      },
    },
  },
  local xlm_mlm_en_2048 = base.Convergence {
    modelName: "hf-glue-xlm-mlm-en-2048",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type xlm \
        --model_name_or_path xlm-mlm-en-2048 \
        --train_batch_size 8 \
        --eval_batch_size 8
      ||| % {common: command_common}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "mnli/acc": {
          success_threshold: {
            fixed_value: 0.80,
          },
          comparison: "greater",
        },
        "mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.80,
          },
          comparison: "greater",
        },
      },
    },
  },
  local distilbert_base_uncased = base.Convergence {
    modelName: "hf-glue-distilbert-b-uc",
    command: utils.scriptCommand(
      |||
        %(common)s --model_type distilbert \
        --model_name_or_path distilbert-base-uncased \
        --train_batch_size 512 \
        --eval_batch_size 512
      ||| % {common: command_common}
    ),
    regressionTestConfig+: {
      metric_success_conditions+: {
        "mnli/acc": {
          success_threshold: {
            fixed_value: 0.65,
          },
          comparison: "greater",
        },
        "mnli-mm/acc": {
          success_threshold: {
            fixed_value: 0.65,
          },
          comparison: "greater",
        },
      },
    },
  },
  local hf_glue = base.PyTorchTest {
    modelName: "hf-glue",
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "nlp-finetuning-ds",
              gcePersistentDisk: {
                pdName: "nlp-finetuning-datasets-us-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "nlp-finetuning-ds",
                readOnly: true,
              }],
              resources+: {
                requests: {
                  cpu: "12.0",
                  memory: "80Gi",
                },
              },
            } for container in super.containers
          ],
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
    hf_glue + v3_8 + bert_base_cased + timeouts.Hours(1),
    hf_glue + v2_8 + bert_base_cased + timeouts.Hours(1),
    hf_glue + v3_8 + xlnet_large_cased + timeouts.Hours(3),
    hf_glue + v3_8 + roberta_large + timeouts.Hours(3),
    hf_glue + v3_8 + xlm_mlm_en_2048 + timeouts.Hours(4),
    hf_glue + v3_8 + distilbert_base_uncased + timeouts.Hours(1),
  ],
}
