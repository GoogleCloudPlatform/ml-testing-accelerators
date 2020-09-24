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
  local gpu_command_base = |||
    unset XRT_TPU_CONFIG
    export GPU_NUM_DEVICES=%(num_gpus)s
    python3 pytorch/xla/test/test_train_mp_imagenet.py \
    --logdir=$(MODEL_DIR) \
    --model=resnet50 \
    --batch_size=128 \
    --log_steps=200 \
    --num_workers=4 \
    --num_epochs=2 \
    --datadir=/datasets/imagenet-mini \
  |||,
  local resnet50_gpu = common.PyTorchTest {
    imageTag: "nightly_3.6_cuda",
    modelName: "resnet50-mp",
    volumeMap+: {
      datasets: common.datasetsVolume
    },
    cpu: "13.0",
    memory: "40Gi",
  },
  local v100 = {
    accelerator: gpus.teslaV100,
    command: utils.scriptCommand(
      gpu_command_base % 1
    ),
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 + { count: 4 },
    command: utils.scriptCommand(
      gpu_command_base % 4
    ),
  },
  configs: [
    resnet50_gpu + v100 + common.Functional + timeouts.Hours(2),
    resnet50_gpu + v100x4 + common.Functional + timeouts.Hours(2),
  ],
}
