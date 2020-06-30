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
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";

{
  local maskrcnn = common.ModelGardenTest {
    modelName: "maskrcnn",
    paramsOverride:: {
      eval: {
        eval_file_pattern: "$(COCO_DIR)/val*",
        batch_size: 8,
        val_json_file: "$(COCO_DIR)/instances_val2017.json",
      },
      train: {
        iterations_per_loop: 5000,
        checkpoint: {
          path: "$(RESNET_PRETRAIN_DIR)/resnet50-checkpoint-2018-02-07",
          prefix: "resnet50/",
        },
        total_steps: error "Must set `train.total_steps`",
        batch_size: error "Must set `train.batch_size`",
        train_file_pattern: "$(COCO_DIR)/train*",
      },
      postprocess: {
        pre_nms_num_boxes: 1000,
      },
    },
    command: [
      "python3",
      "official/vision/detection/main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--strategy_type=tpu",
      "--model=mask_rcnn",
      "--params_override=%s" % (std.manifestYamlDoc(self.paramsOverride) + "\n"),
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local convergence = common.Convergence {
    command+: [
      "--mode=train",
    ],
    paramsOverride+: {
      train+: {
        total_steps: 22500,
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train+: {
        batch_size: 32,
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    paramsOverride+: {
      train+: {
        batch_size: 64,
      },
    },
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    paramsOverride+: {
      train+: {
        batch_size: 128,
      },
    },
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    paramsOverride+: {
      train+: {
        batch_size: 256,
      },
    },
  },

  configs: [
    maskrcnn + convergence + v2_8,
    maskrcnn + convergence + v3_8,
    maskrcnn + convergence + v2_32,
    maskrcnn + convergence + v3_32,
  ],
}

