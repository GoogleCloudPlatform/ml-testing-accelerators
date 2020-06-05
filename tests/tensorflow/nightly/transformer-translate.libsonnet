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
local gpus = import "templates/gpus.libsonnet";

{
  local transformer = common.ModelGardenTest {
    modelName: "transformer-translate",
    command: [
      "python3",
      "official/nlp/transformer/transformer_main.py",
      "--param_set=big",
      "--max_length=64",
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

  local gpu_common = {
    local config = self,

    command+: [
      "--num_gpus=%d" % config.accelerator.number,
      "--steps_between_evals=5000",
    ],
  },
  local k80 = gpu_common {
    accelerator: gpus.teslaK80,
    command+: [
      "--batch_size=2048",
    ],
  },
  local v100 = gpu_common {
    accelerator: gpus.teslaV100,
    command+: [
      "--batch_size=4096",
    ],
  },

  local tpu_common = {
    command+: [
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--distribution_strategy=tpu",
      "--steps_between_evals=10000",
      "--static_batch=true",
      "--use_ctl=true",
      "--padded_decode=true",
      "--decode_batch_size=32",
      "--decode_max_length=97",
    ],
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
    command+: [
      "--batch_size=6144",
    ],
  },
  local v3_8 = tpu_common {
    accelerator: tpus.v3_8,
    command+: [
      "--batch_size=6144",
    ],
  },
  local v2_32 = tpu_common {
    accelerator: tpus.v2_32,
    command+: [
      "--batch_size=24576",
    ],
  },
  local v3_32 = tpu_common {
    accelerator: tpus.v3_32,
    command+: [
      "--batch_size=24576",
    ],
  },

  configs: [
    transformer + k80 + functional + timeouts.Hours(6) + mixins.Experimental,
    transformer + v100 + functional + timeouts.Hours(3),
    transformer + k80 + convergence  + mixins.Experimental,
    transformer + v100 + convergence  + mixins.Experimental,
    transformer + v2_8 + functional,
    transformer + v3_8 + functional,
    transformer + v2_8 + convergence,
    transformer + v3_8 + convergence ,
    transformer + v2_32 + functional,
    transformer + v3_32 + functional,
    transformer + v2_32 + convergence,
    transformer + v3_32 + convergence,
  ],
}
