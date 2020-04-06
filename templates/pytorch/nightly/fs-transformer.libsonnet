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
  local transformer = base.PyTorchTest {
    modelName: "fs-transformer",
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "wmt18-pd",
              gcePersistentDisk: {
                pdName: "wmt18-en-de-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "wmt18-pd",
                readOnly: true,
              }],
              resources+: {
                requests: {
                  cpu: "9.0",
                  memory: "30Gi",
                },
              },
            } for container in super.containers
          ],
        },
      },
    },
  },
  local functional = base.Functional {
    command: utils.scriptCommand(
      |||
        python3 \
          /tpu-examples/deps/fairseq/train.py \
          /datasets/wmt18_en_de_bpej32k \
          --tensorboard-logdir=$(MODEL_DIR) \
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
          --no-save \
          --max-epoch=1 \
          --log_steps=10 \
          --train-subset=valid \
          --valid-subset=test \
          --input_shapes=128x64
      |||
    ),
  },
  local convergence = base.Convergence {
    command: utils.scriptCommand(
      |||
        pip install --editable /tpu-examples/deps/fairseq
        python3 \
          /tpu-examples/deps/fairseq/train.py \
          /datasets/wmt18_en_de_bpej32k  \
          --tensorboard-logdir=$(MODEL_DIR)  \
          --metrics_debug \
          --arch=transformer_vaswani_wmt_en_de_big \
          --max-target-positions=64  \
          --attention-dropout=0.1 \
          --no-progress-bar  \
          --save-interval=5  \
          --save-dir=/tmp/checkpoints \
          --criterion=label_smoothed_cross_entropy \
          --source-lang=en \
          --lr-scheduler=inverse_sqrt \
          --min-lr=1e-09 \
          --skip-invalid-size-inputs-valid-test  \
          --target-lang=de \
          --label-smoothing=0.1  \
          --update-freq=1 \
          --optimizer=adam \
          --adam-betas='(0.9,0.98)' \
          --warmup-init-lr=1e-07 \
          --lr=0.0005 \
          --warmup-updates=4000  \
          --share-all-embeddings \
          --dropout=0.3  \
          --weight-decay=0.0 \
          --num_cores=8  \
          --max-epoch=25 \
          --log_steps=200 \
          --train-subset=train \
          --valid-subset=valid \
          --input_shapes 256x64 512x32 640x16
        bleu=`fairseq-generate \
           /datasets/wmt18_en_de_bpej32k \
           --remove-bpe --quiet --lenpen 0.6 --beam 4 \
           --path /tmp/checkpoints/checkpoint25.pt \
           --skip-invalid-size-inputs-valid-test | grep BLEU \
           | grep -v loadi | tail -1 | cut -d '=' -f 3| cut -d'.' -f 1`
        echo 'BLEU score is' $bleu
        test $bleu -gt 27
      |||
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
    regressionTestConfig+: {
      metric_success_conditions+: {
        "train-wps_final": {
          success_threshold: {
            fixed_value: 11700,
          },
          comparison: "greater",
        },
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    transformer + v3_8 + functional + timeouts.Hours(1),
    transformer + v3_8 + convergence + timeouts.Hours(25),
  ],
}
