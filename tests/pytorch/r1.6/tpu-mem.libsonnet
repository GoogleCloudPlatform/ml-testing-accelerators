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

# "--datadir=/datasets/imagenet-mini",
local common = import "common.libsonnet";
local mixins = import "templates/mixins.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local tpuMem = common.PyTorchTest {
    modelName: "tpu-mem",
    command: utils.scriptCommand(
      |||
        python3 pytorch/xla/test/test_train_mp_imagenet.py --model=resnet50 --num_workers=8 --log_steps=200 --num_epochs=2 --batch_size=400 --fake_data
        echo "\nFinished first round of training.\n"
        python3 pytorch/xla/test/test_train_mp_imagenet.py --model=resnet50 --num_workers=8 --log_steps=200 --num_epochs=2 --batch_size=400 --fake_data
        echo "\nFinished second round of training.\n"
      |||
    ),
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
                  cpu: "8.0",
                  memory: "20Gi",
                },
              },
            },
          },
        },
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    tpuMem + v3_8 + common.Functional + timeouts.Hours(1),
  ],
}
