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
local tpus = import "../../tpus.libsonnet";

{
  local efficientnet = base.LegacyTpuTest {
    modelName: "efficientnet",
    command: [
      "python3",
      "/tpu/models/official/efficientnet/main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--iterations_per_loop=1000",
      "--mode=train",
      "--use_cache=False",
      "--data_dir=$(IMAGENET_DIR)",
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--train_batch_size=2048",
      "--train_steps=218948",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--train_batch_size=2048",
      "--train_steps=218948",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      "--train_batch_size=4096",
      "--train_steps=109474",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      "--train_batch_size=4096",
      "--train_steps=109474",
    ],
  },
  local convergence = base.Convergence,

  configs: [
    efficientnet + v2_8 + convergence,
    efficientnet + v3_8 + convergence,
    efficientnet + v2_32 + convergence,
    efficientnet + v3_32 + convergence,
  ],
}
