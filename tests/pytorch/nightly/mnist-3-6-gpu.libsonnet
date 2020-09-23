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
  local mnist = common.PyTorchTest {
    imageTag: "nightly_3.6_cuda",
    modelName: "mnist-3-6",
    command: utils.scriptCommand(
      |||
        unset XRT_TPU_CONFIG
        export GPU_NUM_DEVICES=1
        python3 pytorch/xla/test/test_train_mp_mnist.py --logdir=$(MODEL_DIR)
      |||
    ),
  },

  local convergence = common.Convergence {
    # Run daily instead of 2x per week since convergence is fast.
    schedule: "35 17 * * *",
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 98.0,
          },
          comparison: "greater",
        },
      },
    },
  },

  local v100 = {
    accelerator: gpus.teslaV100,
  },
  configs: [
    mnist + v100 + convergence + timeouts.Hours(1),
  ],
}
