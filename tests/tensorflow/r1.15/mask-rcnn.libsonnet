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
  local mask_rcnn = common.LegacyTpuTest {
    modelName: "mask-rcnn",
    paramsOverride:: {
      training_file_pattern: "$(COCO_DIR)/train*",
      train_batch_size: error "Must set `train_batch_size`",
      total_steps: 22500,
      checkpoint: "$(RESNET_PRETRAIN_DIR)/resnet50-checkpoint-2018-02-07",
      backbone: "resnet50",
      validation_file_pattern: "$(COCO_DIR)/val*",
      val_json_file: "$(COCO_DIR)/instances_val2017.json",
    },
    command: [
      "python3",
      "/tpu/models/official/mask_rcnn/mask_rcnn_main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--eval_after_training=False",
      "--iterations_per_loop=1000",
      "--mode=train",
      "--data_dir=$(IMAGENET_DIR)",
      "--params_override=%s" % (std.manifestYamlDoc(self.paramsOverride) + "\n"),
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train_batch_size: 32,
    },
    command+: [
      "--num_cores=8",
      "--config_file=/tpu/models/official/mask_rcnn/configs/cloud/v2-8.yaml",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    paramsOverride+: {
      train_batch_size: 32,
    },
    command+: [
      "--num_cores=8",
      "--config_file=/tpu/models/official/mask_rcnn/configs/cloud/v3-8.yaml",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    paramsOverride+: {
      train_batch_size: 128,
    },
    command+: [
      "--num_cores=32",
      "--config_file=/tpu/models/official/mask_rcnn/configs/cloud/v2-32.yaml",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    paramsOverride+: {
      train_batch_size: 128,
    },
    command+: [
      "--num_cores=32",
      "--config_file=/tpu/models/official/mask_rcnn/configs/cloud/v3-32.yaml",
    ],
  },
  local convergence = common.Convergence,
  local functional = common.Functional {
    paramsOverride+: {
      total_steps: 1000,
    },
  },

  configs: [
    mask_rcnn + v2_8 + convergence,
    mask_rcnn + v3_8 + convergence,
    mask_rcnn + v2_32 + convergence + tpus.reserved + {schedule: "10 4 * * 0,2,4"},
    mask_rcnn + v3_32 + convergence,
    mask_rcnn + v2_8 + functional,
    mask_rcnn + v3_8 + functional,
    mask_rcnn + v2_32 + functional,
    mask_rcnn + v3_32 + functional,
  ],
}
