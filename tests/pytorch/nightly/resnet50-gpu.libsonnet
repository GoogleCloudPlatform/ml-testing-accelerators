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

local common = import "common.libsonnet";
local gpus = import "templates/gpus.libsonnet";
local mixins = import "templates/mixins.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local command_common = |||
    unset XRT_TPU_CONFIG
    export GPU_NUM_DEVICES=4
    python3 pytorch/xla/test/test_train_mp_imagenet.py \
    --logdir=$(MODEL_DIR) \
    --model=resnet50 \
    --batch_size=128 \
    --log_steps=200 \
  |||,
  local resnet50_MP = common.PyTorchTest {
    imageTag: "nightly_3.6_cuda",
    modelName: "resnet50-mp",
    volumeMap+: {
      datasets: common.datasetsVolume
    },
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              resources+: {
                requests: {
                  cpu: "13.0",
                  memory: "40Gi",
                },
              },
            },
          },
        },
      },
    },
  },
  local functional = common.Functional {
    command: utils.scriptCommand(
      |||
        %(command_common)s  --num_workers=4 \
          --num_epochs=2 \
          --datadir=/datasets/imagenet-mini \
      ||| % command_common
    ),
    #command: utils.scriptCommand(
    #  |||
    #    %(command_common)s  --no-save \
    #      --max-epoch=1 \
    #      --log_steps=10 \
    #      --train-subset=valid \
    #      --valid-subset=test \
    #      --input_shapes=128x64
    #  ||| % command_common
    #),
  },
  local v100 = {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 + { count: 4 },
  },
  configs: [
    resnet50_MP + v100x4 + functional + timeouts.Hours(2),
  ],
}
