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
local tpus = import "templates/tpus.libsonnet";

{
  local unet = common.LegacyTpuTest {
    modelName: "unet",
    paramsOverride: {
      train_steps: error "Must set `train_steps`",
      train_batch_size: error "Must set `train_batch_size`",
    },
    command: [
      "python3",
      "/tpu/models/official/unet3d/unet_main.py",
      "--iterations_per_loop=100",
      "--mode=train",
      "--training_file_pattern=$(LITS_DIR)/train*",
      "--eval_file_pattern=$(LITS_DIR)/val*",
      "--params_override=%s" % (std.manifestYamlDoc(self.paramsOverride) + "\n"),
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    paramsOverride+: {
      train_steps: 5000,
      train_batch_size: 32,
    },
    command+: [
      "--config_file=/tpu/models/official/unet3d/configs/cloud/v3-8_128x128x128_dice.yaml",
      "--num_cores=8",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    paramsOverride+: {
      train_steps: 1200,
      train_batch_size: 8,
      input_image_size: [128,128,128],
    },
    command+: [
      "--config_file=/tpu/models/official/unet3d/configs/cloud/v3-32_256x256x256_ce.yaml",
      "--num_cores=32",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    paramsOverride+: {
      train_steps: 1200,
      train_batch_size: 128,
    },
    command+: [
      "--config_file=/tpu/models/official/unet3d/configs/cloud/v3-8_128x128x128_dice.yaml",
      "--num_cores=32",
    ],
  },
  local convergence = common.Convergence,
  local functional = common.Functional {
    paramsOverride+: {
      train_steps: 100,
    },
  },
  local reserved = {
    tpuSettings+: {
      reserved: "true",
    },
  },

  configs: [
    unet + v3_8 + convergence,
    unet + v2_32 + convergence + reserved + {schedule: "0 1 * * 1,3,5,6"},
    unet + v3_32 + convergence,
    unet + v3_8 + functional,
    unet + v2_32 + functional,
    unet + v3_32 + functional,
  ],
}
