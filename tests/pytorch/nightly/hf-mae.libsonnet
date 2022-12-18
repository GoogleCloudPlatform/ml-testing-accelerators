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
    pip install datasets evaluate sklearn
    pip install -r examples/pytorch/_tests_requirements.txt
  |||,
  local command_copy_metrics = |||
    gsutil -m cp -r ./tensorboard-metrics/* $(MODEL_DIR)
  |||,
  local hf_mae = common.PyTorchTest {
    local config = self,
    modelName: 'hf-mae',
    paramsOverride:: {
      scriptPath: 'examples/pytorch/xla_spawn.py',
      tpuCores: config.accelerator.numCores,
      trainCommand: [
        'python3',
        self.scriptPath,
        '--num_cores=%d' % config.paramsOverride.tpuCores,
        'examples/pytorch/image-pretraining/run_mae.py',
        '--model_name=%s' % config.paramsOverride.model_name,
        '--dataset_name=cifar10',
        '--remove_unused_columns=False',
        '--label_names=pixel_values',
        '--mask_ratio=0.75',
        '--norm_pix_loss=True',
        '--do_train=true',
        '--do_eval=true',
        '--base_learning_rate=1.5e-4',
        '--lr_scheduler_type=cosine',
        '--weight_decay=0.05',
        '--num_train_epochs=3',
        '--warmup_ratio=0.05',
        '--per_device_train_batch_size=%d ' % config.paramsOverride.per_device_train_batch_size,
        '--per_device_eval_batch_size=%d ' % config.paramsOverride.per_device_eval_batch_size,
        '--logging_strategy=steps',
        '--logging_steps=30',
        '--evaluation_strategy=epoch',
        '--save_strategy=epoch',
        '--load_best_model_at_end=True',
        '--save_total_limit=3',
        '--seed=1337',
        '--output_dir=MAE',
        '--overwrite_output_dir=true',
        '--logging_dir=./tensorboard-metrics',
        '--task_name=MAE',
        '--overwrite_cache=true',
        '--tpu_metrics_debug=true',
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
  local hf_vit_mae = common.Convergence {
    modelName: 'hf-vit-mae',
    paramsOverride+:: {
      model_name: 'vit-mae',
      per_device_train_batch_size: 8,
      per_device_eval_batch_size: 8,
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
  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_BF16=$(XLA_USE_BF16)
      |||,
      tpuVmExtraSetup: |||
        echo 'export XLA_USE_BF16=1' >> ~/.bash_profile
      |||,
    },
  },
  local pjrt = tpuVm + experimental.PjRt {
    modelName: 'hf-mae-pjrt',
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    hf_mae + v2_8 + hf_vit_mae + timeouts.Hours(2),
    hf_mae + v3_8 + hf_vit_mae + timeouts.Hours(2),
    hf_mae + v4_8 + hf_vit_mae + timeouts.Hours(2),
  ],
}
