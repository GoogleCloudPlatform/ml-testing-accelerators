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

local experimental = import '../experimental.libsonnet';
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
    sed '/torchvision/d' examples/pytorch/_tests_requirements.txt > no_vision_require.txt
    pip install -r no_vision_require.txt
  |||,
  local command_copy_metrics = |||
    gsutil -m cp -r ./tensorboard-metrics/* $(MODEL_DIR)
  |||,
  local bert_base_cased = self.bert_base_cased,
  bert_base_cased:: common.Convergence {
    modelName: 'hf-glue-bert-b-c',
    paramsOverride+:: {
      model_name_or_path: 'bert-base-cased',
      per_device_train_batch_size: 64,
      per_device_eval_batch_size: 64,
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/accuracy': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.80,
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
  local hf_glue = self.hf_glue,
  hf_glue:: common.PyTorchTest {
    local config = self,
    modelName: 'hf-glue',
    paramsOverride:: {
      scriptPath: 'examples/pytorch/xla_spawn.py',
      tpuCores: config.accelerator.numCores,
      trainCommand: [
        'python3',
        self.scriptPath,
        '--num_cores=%d' % config.paramsOverride.tpuCores,
        'examples/pytorch/text-classification/run_glue.py',
        '--model_name_or_path=%s' % config.paramsOverride.model_name_or_path,
        '--logging_dir=./tensorboard-metrics',
        '--task_name=MNLI',
        '--cache_dir=./cache_dir',
        '--do_train=true',
        '--do_eval=true',
        '--num_train_epochs=3',
        '--max_seq_length=128',
        '--learning_rate=3e-5',
        '--output_dir=MNLI',
        '--overwrite_output_dir=true',
        '--logging_steps=30',
        '--save_steps=3000',
        '--overwrite_cache=true',
        '--debug=tpu_metrics_debug',
        '--per_device_train_batch_size=%d ' % config.paramsOverride.per_device_train_batch_size,
        '--per_device_eval_batch_size=%d ' % config.paramsOverride.per_device_eval_batch_size,
      ],
    },
    command: utils.scriptCommand(
      |||
        %s
        %s
        %s
      ||| % [
        command_common,
        utils.toCommandString(self.paramsOverride.trainCommand),
        command_copy_metrics,
      ]
    ),
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
  local xlnet_large_cased = self.xlnet_large_cased,
  xlnet_large_cased:: common.Convergence {
    modelName: 'hf-glue-xlnet-l-c',
    paramsOverride+:: {
      model_name_or_path: 'xlnet-large-cased',
      per_device_train_batch_size: 32,
      per_device_eval_batch_size: 16,
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/accuracy': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.85,
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
  local roberta_large = self.roberta_large,
  roberta_large:: common.Convergence {
    modelName: 'hf-glue-roberta-l',
    paramsOverride+:: {
      model_name_or_path: 'roberta-large',
      per_device_train_batch_size: 16,
      per_device_eval_batch_size: 16,
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/accuracy': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.85,
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
  local distilbert_base_uncased = self.distilbert_base_uncased,
  distilbert_base_uncased:: common.Convergence {
    modelName: 'hf-glue-distilbert-b-uc',
    paramsOverride+:: {
      model_name_or_path: 'distilbert-base-uncased',
      per_device_train_batch_size: 512,
      per_device_eval_batch_size: 512,
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/accuracy': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.70,
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
  local tpuVm = {
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_BF16=$(XLA_USE_BF16)
      |||,
    },
  },
  // TODO: Remove after 2.1 release cut
  local xrt = self.xrt,
  xrt:: common.XrtTpuVmMixin + tpuVm,
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin + tpuVm {
    modelName: 'hf-glue-pjrt',
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
    hf_glue + v2_8 + bert_base_cased + timeouts.Hours(2) + xrt,
    hf_glue + v3_8 + xlnet_large_cased + timeouts.Hours(5) + xrt,
    hf_glue + v3_8 + roberta_large + timeouts.Hours(8) + xrt,
    hf_glue + v3_8 + distilbert_base_uncased + timeouts.Hours(2) + xrt,
    hf_glue + v4_8 + roberta_large + timeouts.Hours(6) + xrt,
    hf_glue + v4_8 + distilbert_base_uncased + timeouts.Hours(2) + pjrt,
  ],
}
