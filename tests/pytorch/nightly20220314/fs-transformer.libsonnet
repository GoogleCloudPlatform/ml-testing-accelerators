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
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local transformer = common.PyTorchTest {
    local config = self,

    modelName: 'fs-transformer',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    paramsOverride:: {
      scriptPath: 'tpu-examples/deps/fairseq/train.py',
      logSteps: 200,
      trainSubset: 'train',
      validSubset: 'valid',
      inputShape: ['256x64', '512x32'],
      trainCommand: [
        'python3',
        self.scriptPath,
        '/datasets/wmt18_en_de_bpej32k',
        '--metrics_debug',
        '--arch=transformer_vaswani_wmt_en_de_big',
        '--max-target-positions=64',
        '--attention-dropout=0.1',
        '--no-progress-bar',
        '--criterion=label_smoothed_cross_entropy',
        '--source-lang=en',
        '--lr-scheduler=inverse_sqrt',
        '--min-lr=1e-09',
        '--skip-invalid-size-inputs-valid-test',
        '--target-lang=de',
        '--label-smoothing=0.1',
        '--update-freq=1',
        '--optimizer=adam',
        '--adam-betas',
        '(0.9,0.98)',
        '--warmup-init-lr=1e-07',
        '--lr=0.0005',
        '--warmup-updates=4000',
        '--share-all-embeddings',
        '--dropout=0.3',
        '--weight-decay=0.0',
        '--num_cores=%d' % config.accelerator.numCores,
        '--log_steps=%d' % config.paramsOverride.logSteps,
        '--train-subset=%s' % config.paramsOverride.trainSubset,
        '--valid-subset=%s' % config.paramsOverride.validSubset,
        '--input_shapes',
      ] + config.paramsOverride.inputShape,
    },
    cpu: '9.0',
    memory: '30Gi',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap:: {},
        },
      },
    },
  },

  // Run the test over a small subset of the data.
  local base_functional = common.Functional {
    paramsOverride+:: {
      logSteps: 10,
      inputShape: ['128x64'],
    },
  },
  local checkpoint_local = base_functional {
    modelName: 'fs-checkpoint-local',
    paramsOverride+:: {
      trainSubset: 'test',
      trainCommand+: [
        '--save-interval=1',
        '--save-dir=/tmp/checkpoints',
      ],
    },
    command: utils.scriptCommand(
      |||
        %s
        %s
      ||| % [
        utils.toCommandString(self.paramsOverride.trainCommand + ['--max-epoch=1']),
        utils.toCommandString(self.paramsOverride.trainCommand + ['--max-epoch=2']),
      ]
    ),
  },
  local checkpoint_gcs = base_functional {
    modelName: 'fs-checkpoint-gcs',
    paramsOverride+:: {
      trainSubset: 'test',
      trainCommand+: [
        '--save-interval=1',
        '--save-dir=$(MODEL_DIR)/checkpoints',
      ],
    },
    command: utils.scriptCommand(
      |||
        %s
        set +e
        %s
        gsutil ls -l $(MODEL_DIR)/checkpoints
        gsutil rm -r $(MODEL_DIR)/checkpoints
      ||| % [
        utils.toCommandString(self.paramsOverride.trainCommand + ['--max-epoch=1']),
        utils.toCommandString(self.paramsOverride.trainCommand + ['--max-epoch=2']),
      ]
    ),
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              XLA_USE_BF16: '1',
            },
          },
        },
      },
    },
  },
  local functional_no_save = base_functional {
    local config = self,
    paramsOverride+:: {
      trainSubset: 'valid',
      validSubset: 'test',
      trainCommand+: [
        '--no-save',
        '--max-epoch=1',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },

  local convergence = common.Convergence {
    paramsOverride+:: {
      trainCommand+: [
        '--save-interval=5',
        '--save-dir=/tmp/checkpoints',
        '--max-epoch=25',
      ],
    },
    command: utils.scriptCommand(
      |||
        pip install --editable tpu-examples/deps/fairseq
        %s 2>&1 | tee training_logs.txt
        bleu=`fairseq-generate \
          /datasets/wmt18_en_de_bpej32k \
          --remove-bpe --quiet --lenpen 0.6 --beam 4 \
          --path /tmp/checkpoints/checkpoint25.pt \
          --skip-invalid-size-inputs-valid-test | grep BLEU \
          | grep -v loadi | tail -1 | cut -d '=' -f 3| cut -d'.' -f 1`
        echo 'BLEU score is' $bleu
        wps=$(cat training_logs.txt | grep '| wps ' | tail -1 | grep -o -E ' wps [0-9]+' | sed 's/[^0-9]*//g')
        echo 'final words per second (wps) is' $wps
        test $bleu -gt 27 -a $wps -gt 10000
      ||| % utils.toCommandString(self.paramsOverride.trainCommand)
    ),
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              XLA_USE_BF16: '1',
            },
          },
        },
      },
    },
  },

  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExtraSetup: |||
        git clone --recursive https://github.com/pytorch-tpu/examples.git tpu-examples/
        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile
        echo 'export XLA_USE_BF16=1' >> ~/.bash_profile
        %s
      ||| % common.tpu_vm_latest_install,
    },
  },

  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  configs: [
    transformer + v3_8 + functional_no_save + timeouts.Hours(25) + tpuVm,
    transformer + v3_32 + functional_no_save + timeouts.Hours(1) + tpuVm,
  ],
}
