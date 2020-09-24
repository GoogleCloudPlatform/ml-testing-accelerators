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

  local gpu_command_base = |||
    unset XRT_TPU_CONFIG
    export GPU_NUM_DEVICES=%(num_gpus)s
    python3 pytorch/xla/test/test_train_mp_mnist.py --logdir=$(MODEL_DIR)
  |||,

  local mnist_gpu = common.PyTorchTest {
    imageTag: "nightly_3.6_cuda",
    modelName: "mnist",
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
    # Run daily instead of 2x per week since convergence is fast.
    schedule: "0 17 * * *",
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
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
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
    mnist + v2_8 + convergence + timeouts.Hours(1),
    mnist + v3_8 + convergence + timeouts.Hours(1),
    mnist_pod + v2_32 + convergence,
    mnist_pod + v3_32 + convergence,
    mnist_gpu + v100 + convergence + timeouts.Hours(1),
    mnist_gpu + v100x4 + convergence + timeouts.Hours(1),
  ],
}
