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
local tpus = import "templates/tpus.libsonnet";

{
  local retinanet = base.LegacyTpuTest {
    modelName: "retinanet",
    paramsOverride: {
      train: {
        train_batch_size: "Must set `train_batch_size`",
        total_steps: 22500,
        iterations_per_loop: 1000,
        train_file_pattern: "$(COCO_DIR)/train*",
        checkpoint: {
          path: "$(RESNET_PRETRAIN_DIR)/resnet50-checkpoint-2018-02-07",
          prefix: "resnet50/"
        },
      },
    },
    command: [
      "python3",
      "/tpu/models/official/detection/main.py",
      "--mode=train",
      "--use_tpu=True",
      "--params_override=%s" % (std.manifestYamlDoc(self.paramsOverride) + "\n"),
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train+: {
        train_batch_size: 64,
      },
    },
    command+: [
      "--num_cores=8",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    paramsOverride+: {
      train+: {
        train_batch_size: 64,
      },
    },
    command+: [
      "--num_cores=8",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    paramsOverride+: {
      train+: {
        train_batch_size: 256,
      },
    },
    command+: [
      "--num_cores=32",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    paramsOverride+: {
      train+: {
        train_batch_size: 256,
      },
    },
    command+: [
      "--num_cores=32",
    ],
  },
  local convergence = base.Convergence,

  configs: [
    retinanet + v2_8 + convergence,
    retinanet + v3_8 + convergence,
    retinanet + v2_32 + convergence,
    retinanet + v3_32 + convergence,
  ],
}
