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
  local transformer = base.LegacyTpuTest {
    modelName: "transformer",
    command: [
      "t2t-trainer",
      "--model=transformer",
      "--hparams_set=transformer_packed_tpu",
      "--problem=translate_ende_wmt32k_packed",
      "--use_tpu=True",
      "--schedule=train",
      "--data_dir=$(T2T_TRANSFORMER_DIR)",
      "--cloud_tpu_name=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--output_dir=$(MODEL_DIR)",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--iterations_per_loop=100",
      "--tpu_num_shards=8",
      "--train_steps=250000",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--iterations_per_loop=100",
      "--tpu_num_shards=8",
      "--train_steps=250000",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      "--iterations_per_loop=5000",
      "--tpu_num_shards=32",
      "--train_steps=62500",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      "--iterations_per_loop=5000",
      "--tpu_num_shards=32",
      "--train_steps=62500",
    ],
  },
  local convergence = base.Convergence,

  configs: [
    transformer + v2_8 + convergence,
    transformer + v3_8 + convergence,
    transformer + v2_32 + convergence,
    transformer + v3_32 + convergence,
  ],
}
