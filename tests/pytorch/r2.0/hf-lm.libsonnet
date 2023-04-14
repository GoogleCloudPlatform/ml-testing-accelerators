// Copyright 2020 Google LLC
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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local command_common = |||
    git clone https://github.com/huggingface/transformers.git
    cd transformers && pip install .
    git log -1
    pip install datasets evaluate scikit-learn
    python3 examples/pytorch/xla_spawn.py \
      --num_cores 8 \
      examples/pytorch/language-modeling/run_mlm.py \
      --logging_dir ./tensorboard-metrics \
      --cache_dir ./cache_dir \
      --dataset_name wikitext \
      --dataset_config_name wikitext-2-raw-v1 \
      --do_train \
      --do_eval \
      --overwrite_output_dir \
      --output_dir language-modeling \
      --logging_steps 30 \
      --save_steps 3000 \
      --overwrite_cache \
      --tpu_metrics_debug \
  |||,
  local command_copy_metrics = |||
    gsutil -m cp -r ./tensorboard-metrics/* $(MODEL_DIR)
  |||,
  local roberta_base_fine = self.roberta_base_fine,
  roberta_base_fine:: common.Convergence {
    modelName: 'hf-mlm-roberta-b-fine',
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=roberta \
        --model_name_or_path roberta-base \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8
        %(common_copy)s
      ||| % { common: command_common, common_copy: command_copy_metrics }
    ),
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/loss': {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 2.7,
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
  local bert_base_fine = self.bert_base_fine,
  bert_base_fine:: common.Convergence {
    modelName: 'hf-mlm-bert-b-fine',
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=bert \
        --model_name_or_path bert-base-cased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16
        %(common_copy)s
      ||| % { common: command_common, common_copy: command_copy_metrics }
    ),
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/loss': {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 2.8,
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
  local roberta_base_pre = self.roberta_base_pre,
  roberta_base_pre:: common.Convergence {
    modelName: 'hf-mlm-roberta-b-pre',
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=roberta \
        --tokenizer=roberta-base \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8
        %(common_copy)s
      ||| % { common: command_common, common_copy: command_copy_metrics }
    ),
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/loss': {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 7.2,
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
  local bert_base_pre = self.bert_base_pre,
  bert_base_pre:: common.Convergence {
    modelName: 'hf-mlm-bert-b-pre',
    command: utils.scriptCommand(
      |||
        %(common)s --model_type=bert \
        --tokenizer=bert-base-cased \
        --num_train_epochs 5 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16
        %(common_copy)s
      ||| % { common: command_common, common_copy: command_copy_metrics }
    ),
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/loss': {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 7.4,
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
  local hf_lm = self.hf_lm,
  hf_lm:: common.PyTorchTest {
    modelName: 'hf-lm',
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: '12.0',
                memory: '80Gi',
              },
            },
          },
        },
      },
    },
  },
  local tpuVm = self.tpuVm,
  tpuVm:: common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_BF16=$(XLA_USE_BF16)
      |||,
      tpuVmExtraSetup: |||
        pip install tensorboardX google-cloud-storage
        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile
        echo 'export XLA_USE_BF16=1' >> ~/.bash_profile
      |||,
    },
  },
  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  configs: [
    hf_lm + v2_8 + roberta_base_pre + timeouts.Hours(5) + tpuVm,
    hf_lm + v3_8 + roberta_base_fine + timeouts.Hours(3) + tpuVm,
    hf_lm + v3_8 + bert_base_pre + timeouts.Hours(6) + tpuVm,
    hf_lm + v3_8 + bert_base_fine + timeouts.Hours(5) + tpuVm,
  ],
}
