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
  local roberta = {
    modelName: 'roberta-pre',
    paramsOverride: {
      maxEpoch: error 'Must set `maxEpoch`',
      wpsTarget: error 'Must set `wpsTarget`',
    },
    command: utils.scriptCommand(
      |||
        pip install --editable tpu-examples/deps/fairseq
        python3 \
          /tpu-examples/deps/fairseq/train.py \
          /datasets/wikitext-103 \
          --task=masked_lm --criterion=masked_lm \
          --arch=roberta_base --sample-break-mode=complete \
          --tokens-per-sample=512 \
          --optimizer=adam \
          --adam-betas='(0.9,0.98)' \
          --adam-eps=1e-6 \
          --clip-norm=0.0 \
          --lr-scheduler=polynomial_decay \
          --lr=0.0005 \
          --warmup-updates=10000 \
          --dropout=0.1 \
          --attention-dropout=0.1 \
          --weight-decay=0.01 \
          --update-freq=16 \
          --log-format=simple \
          --train-subset=train \
          --valid-subset=valid \
          --num_cores=8 \
          --metrics_debug \
          --save-dir=checkpoints \
          --log_steps=30 \
          --skip-invalid-size-inputs-valid-test \
          --suppress_loss_report \
          --input_shapes 16x512 18x480 21x384 \
          --max-epoch=%(maxEpoch)d \
          2>&1 | tee training_logs.txt
        wps=$(cat training_logs.txt | grep '| wps ' | tail -1 | grep -o -E ' wps [0-9]+' | sed 's/[^0-9]*//g')
        echo 'final words per second (wps) is' $wps
        test $wps -gt %(wpsTarget)d
      ||| % self.paramsOverride,
    ),
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: '9.0',
                memory: '30Gi',
                'ephemeral-storage': '10Gi',
              },
            },
          },
        },
      },
    },
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
    paramsOverride: {
      maxEpoch: 1,
      wpsTarget: 10000,
    },
  },
  local convergence = common.Convergence {
    paramsOverride: {
      maxEpoch: 5,
      wpsTarget: 17000,
    },
  },
  local roberta_tpu_vm = common.PyTorchTest {
    frameworkPrefix: 'pt-nightly',
    modelName: 'roberta-pre',
    schedule: '30 13 * * *',
    command: utils.scriptCommand(
      |||
        %(command_common)s
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        pip3 install --editable examples/deps/fairseq
        python3 \
          examples/deps/fairseq/train.py \
          /datasets/wikitext-103 \
          --task=masked_lm --criterion=masked_lm \
          --arch=roberta_base --sample-break-mode=complete \
          --tokens-per-sample=512 \
          --optimizer=adam \
          --adam-betas='(0.9,0.98)' \
          --adam-eps=1e-6 \
          --clip-norm=0.0 \
          --lr-scheduler=polynomial_decay \
          --lr=0.0005 \
          --warmup-updates=10000 \
          --dropout=0.1 \
          --attention-dropout=0.1 \
          --weight-decay=0.01 \
          --update-freq=16 \
          --log-format=simple \
          --train-subset=train \
          --valid-subset=valid \
          --num_cores=8 \
          --metrics_debug \
          --save-dir=checkpoints \
          --log_steps=30 \
          --skip-invalid-size-inputs-valid-test \
          --suppress_loss_report \
          --input_shapes 16x512 18x480 21x384 \
          --max-epoch=5 \
          2>&1 | tee training_logs.txt
        wps=$(cat training_logs.txt | grep '| wps ' | tail -1 | grep -o -E ' wps [0-9]+' | sed 's/[^0-9]*//g')
        echo 'final words per second (wps) is' $wps
        test $wps -gt 19000
      ||| % common.tpu_vm_nightly_install
    ),
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  configs: [
    common.PyTorchGkePodTest + roberta + v3_32 + functional + timeouts.Hours(1),
    common.PyTorchTest + roberta + v3_8 + functional + timeouts.Hours(1),
    common.PyTorchTest + roberta + v3_8 + convergence + timeouts.Hours(2),
    roberta_tpu_vm + v3_8 + common.Convergence + timeouts.Hours(6) + experimental.PyTorchTpuVmMixin,
  ],
}
