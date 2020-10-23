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
local mixins = import "templates/mixins.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local mnist = common.PyTorchTest {
    modelName: "mnist",
    command: [
      "python3",
      "pytorch/xla/test/test_train_mp_mnist.py",
      "--logdir=$(MODEL_DIR)",
    ],
  },

  local mnist_pod = common.PyTorchXlaDistPodTest {
    modelName: "mnist",
    command: [
      "python3",
      "/usr/share/torch-xla-nightly/pytorch/xla/test/test_train_mp_mnist.py",
      "--logdir=$(MODEL_DIR)",
    ],
  },

  local convergence = common.Convergence {
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

  local v2_8 = {
    accelerator: tpus.v2_8,
    schedule: "0 21 * * *",
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    schedule: "2 21 * * *",
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    schedule: "4 21 * * *",
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    schedule: "6 21 * * *",
  },
  configs: [
    mnist + convergence + v2_8 + timeouts.Hours(1),
    mnist + convergence + v3_8 + timeouts.Hours(1),
    mnist_pod + convergence + v2_32,
    mnist_pod + convergence + v3_32 ,
  ],
}
