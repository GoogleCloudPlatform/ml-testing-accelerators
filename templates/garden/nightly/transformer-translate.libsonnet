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
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local transformer = base.GardenTest {
    modelName: "transformer-translate",
    command: [
      "python3",
      "official/nlp/transformer/transformer_main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--steps_between_evals=10000",
      "--static_batch=true",
      "--use_ctl=true",
      "--param_set=big",
      "--max_length=64",
      "--decode_batch_size=32",
      "--decode_max_length=97",
      "--padded_decode=true",
      "--distribution_strategy=tpu",
      "--data_dir=$(TRANSFORMER_DIR)",
      "--vocab_file=$(TRANSFORMER_DIR)/vocab.ende.32768",
      "--bleu_source=$(TRANSFORMER_DIR)/newstest2014.en",
      "--bleu_ref=$(TRANSFORMER_DIR)/newstest2014.de",
      "--enable_tensorboard",
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--train_steps=20000",
    ],
  },
  local convergence = mixins.Convergence {
    command+: [
      "--train_steps=200000",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--batch_size=6144",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--batch_size=6144",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      "--batch_size=24576",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      "--batch_size=24576",
    ],
  },
  configs: [
    transformer + functional + v2_8,
    transformer + functional + v3_8,
    transformer + convergence + v2_8,
    transformer + convergence + v3_8,
    transformer + functional + v2_32,
    transformer + functional + v3_32,
    transformer + convergence + v2_32,
    transformer + convergence + v3_32,
  ],
}
