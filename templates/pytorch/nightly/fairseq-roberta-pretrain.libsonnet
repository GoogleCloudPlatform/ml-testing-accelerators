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

{
  local base_command = "pip install --editable tpu-examples/deps/fairseq && python3 /tpu-examples/deps/fairseq/train.py /datasets/wikitext-103 --tensorboard-logdir=$(MODEL_DIR) --task=masked_lm --criterion=masked_lm --arch=roberta_base --sample-break-mode=complete --tokens-per-sample=512 --optimizer=adam --adam-betas='(0.9,0.98)' --adam-eps=1e-6 --clip-norm=0.0 --lr-scheduler=polynomial_decay --lr=0.0005 --warmup-updates=10000 --dropout=0.1 --attention-dropout=0.1 --weight-decay=0.01 --update-freq=16 --log-format=simple --train-subset=train --valid-subset=valid --num_cores=8 --metrics_debug --save-dir=checkpoints --log_steps=30 --skip-invalid-size-inputs-valid-test --suppress_loss_report --input_shapes 16x512 18x480 21x384",

  local roberta = base.PyTorchTest {
    modelName: "fairseq-roberta-pretrain",
    command: [
      "/bin/bash",
      "-c",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "wikitext-pd",
              gcePersistentDisk: {
                pdName: "wikitext-103-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "wikitext-pd",
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
    regressionTestConfig: null,
    command+: [
      base_command + " --max-epoch=1",
    ],
  },
  local convergence = base.Convergence {
    command+: [
      base_command + " --max-epoch=5",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    roberta + v3_8 + functional + timeouts.Hours(1),
    roberta + v3_8 + convergence + timeouts.Hours(2),
  ],
}
