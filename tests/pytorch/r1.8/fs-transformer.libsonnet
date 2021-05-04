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
    python3 \
      /tpu-examples/deps/fairseq/train.py \
      /datasets/wmt18_en_de_bpej32k \
      --metrics_debug \
      --arch=transformer_vaswani_wmt_en_de_big \
      --max-target-positions=64 \
      --attention-dropout=0.1 \
      --no-progress-bar \
      --criterion=label_smoothed_cross_entropy \
      --source-lang=en \
      --lr-scheduler=inverse_sqrt  \
      --min-lr=1e-09 \
      --skip-invalid-size-inputs-valid-test \
      --target-lang=de \
      --label-smoothing=0.1 \
      --update-freq=1 \
      --optimizer=adam \
      --adam-betas='(0.9,0.98)' \
      --warmup-init-lr=1e-07 \
      --lr=0.0005 \
      --warmup-updates=4000 \
      --share-all-embeddings \
      --dropout=0.3 \
      --weight-decay=0.0 \
      --num_cores=8 \
  |||,
  local chpt_command_common = |||
    %(command_common)s  --log_steps=10 \
      --train-subset=test \
      --valid-subset=valid \
      --save-interval=1 \
      --input_shapes=128x64 \
  ||| % command_common,
  local transformer = {
    modelName: 'fs-transformer',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '9.0',
    memory: '30Gi',
    regressionTestConfig: {
      metric_subset_to_alert: [
        'total_wall_time',
      ],
      metric_success_conditions: {
        total_wall_time: {
          comparison: 'less',
          success_threshold: {
            stddevs_from_mean: 5,
          },
          wait_for_n_points_of_history: 10,
        },
      },
    },
  },
  local checkpoint_local = common.Functional {
    modelName: 'fs-checkpoint-local',
    command: utils.scriptCommand(
      |||
        %(common)s  --max-epoch=1 \
          --save-dir=/tmp/checkpoints
        %(common)s  --max-epoch=2 \
          --save-dir=/tmp/checkpoints
      ||| % { common: chpt_command_common }
    ),
  },
  local checkpoint_gcs = common.Functional {
    modelName: 'fs-checkpoint-gcs',
    command: utils.scriptCommand(
      |||
        %(common)s  --max-epoch=1 \
          --save-dir=%(savedir)s
        set +e
        %(common)s  --max-epoch=2 \
          --save-dir=%(savedir)s
        gsutil ls -l %(savedir)s
        gsutil rm -r %(savedir)s
      ||| % { common: chpt_command_common, savedir: '$MODEL_DIR/checkpoints' }
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
  local functional_xla_dist = common.Functional {
    condaEnv: 'torch-xla-1.8',
    command: [
      'python',
      '/usr/share/torch-xla-1.8/tpu-examples/deps/fairseq/train.py',
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
      "--adam-betas='(0.9,0.98)'",
      '--warmup-init-lr=1e-07',
      '--lr=0.0005',
      '--warmup-updates=4000',
      '--share-all-embeddings',
      '--dropout=0.3',
      '--weight-decay=0.0',
      '--num_cores=8',
      '--no-save',
      '--max-epoch=1',
      '--log_steps=10',
      '--train-subset=valid',
      '--valid-subset=test',
      '--input_shapes=128x64',
    ],
  },
  local functional = common.Functional {
    command: utils.scriptCommand(
      |||
        %(command_common)s  --no-save \
          --max-epoch=1 \
          --log_steps=10 \
          --train-subset=valid \
          --valid-subset=test \
          --input_shapes=128x64
      ||| % command_common
    ),
  },
  local convergence = common.Convergence {
    command: utils.scriptCommand(
      |||
        pip install --editable /tpu-examples/deps/fairseq
        %(command_common)s  --save-interval=5 \
          --save-dir=/tmp/checkpoints \
          --max-epoch=25 \
          --log_steps=200 \
          --train-subset=train \
          --valid-subset=valid \
          --input_shapes 256x64 512x32 \
          2>&1 | tee training_logs.txt
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
      ||| % command_common
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
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  configs: [
    common.PyTorchXlaDistPodTest + transformer + v3_32 + functional_xla_dist + timeouts.Hours(1),
    common.PyTorchTest + transformer + v3_8 + functional + timeouts.Hours(1),
    common.PyTorchTest + transformer + v3_8 + convergence + timeouts.Hours(25),
    common.PyTorchTest + transformer + v3_8 + checkpoint_local + timeouts.Hours(2),
    common.PyTorchTest + transformer + v3_8 + checkpoint_gcs + timeouts.Hours(2),
  ],
}
