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
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";

{
  local cifar = base.PyTorchTest {
    modelName: "cifar-tv",
    command: [
      "python3",
      "pytorch/xla/test/test_train_cifar.py",
      "--logdir=$(MODEL_DIR)",
      "--use_torchvision=True",
      "--metrics_debug",
      "--target_accuracy=72",
    ],
  },
  local convergence = base.Convergence {
    # Run at 6AM PST daily instead of 2x per week since convergence is fast.
    schedule: "0 14 * * *",
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 72.0,
          },
          comparison: "greater",
        },
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    cifar + v2_8 + convergence + timeouts.Hours(1),
    cifar + v3_8 + convergence + timeouts.Hours(1),
  ],
}
