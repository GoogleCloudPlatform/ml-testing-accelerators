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
  local roberta = common.PyTorchTest {
    local config = self,

    modelName: 'roberta-pre',
    paramsOverride:: {
      maxEpoch: error 'Must set `maxEpoch`',
      wpsTarget: error 'Must set `wpsTarget`',
      scriptPath: 'tpu-examples/deps/fairseq/train.py',
      inputShape: ['16x512', '18x480', '21x384'],
      trainCommand: [
        'python3',
        self.scriptPath,
        '/datasets/wikitext-103',
        '--task=masked_lm',
        '--criterion=masked_lm',
        '--arch=roberta_base',
        '--sample-break-mode=complete',
        '--tokens-per-sample=512',
        '--optimizer=adam',
        '--adam-betas',
        '(0.9,0.98)',
        '--adam-eps=1e-6',
        '--clip-norm=0.0',
        '--lr-scheduler=polynomial_decay',
        '--lr=0.0005',
        '--warmup-updates=10000',
        '--dropout=0.1',
        '--attention-dropout=0.1',
        '--weight-decay=0.01',
        '--update-freq=16',
        '--log-format=simple',
        '--train-subset=train',
        '--valid-subset=valid',
        '--num_cores=8',
        '--metrics_debug',
        '--save-dir=checkpoints',
        '--log_steps=30',
        '--skip-invalid-size-inputs-valid-test',
        '--suppress_loss_report',
        '--max-epoch=%d' % self.maxEpoch,
        '--input_shapes',
      ] + config.paramsOverride.inputShape,

    },
    command: utils.scriptCommand(
      |||
        pip install --editable tpu-examples/deps/fairseq
        %(command_string)s 2>&1 | tee training_logs.txt
        wps=$(cat training_logs.txt | grep '| wps ' | tail -1 | grep -o -E ' wps [0-9]+' | sed 's/[^0-9]*//g')
        echo 'final words per second (wps) is' $wps
        test $wps -gt %(wpsTarget)d
      ||| % {
        command_string: utils.toCommandString(config.paramsOverride.trainCommand),
        wpsTarget: config.paramsOverride.wpsTarget,
      }
    ),
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '9.0',
    memory: '30Gi',
    'ephemeral-storage': '10Gi',
  },
  local functional = common.Functional {
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap:: {},
        },
        literals: {},
      },
    },
    paramsOverride+: {
      maxEpoch: 1,
      wpsTarget: 10000,
    },
  },
  local convergence = common.Convergence {
    paramsOverride+: {
      maxEpoch: 5,
      wpsTarget: 19000,
    },
  },
  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExtraSetup: |||
        git clone --recursive https://github.com/pytorch-tpu/examples.git tpu-examples/
        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile
        echo 'export XLA_USE_BF16=1' >> ~/.bash_profile
      |||,
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  configs: [
    roberta + v3_8 + functional + timeouts.Hours(1),
    roberta + v3_8 + convergence + timeouts.Hours(6) + tpuVm,
  ],
}
