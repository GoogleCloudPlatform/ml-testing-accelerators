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
  local retinanet = base.GardenTest {
    modelName: "retinanet",
    paramsOverride:: {
      eval: {
        eval_file_pattern: "gs://xl-ml-test-us-central1/data/coco/val*",
        batch_size: 8,
        val_json_file: "gs://xl-ml-test-us-central1/data/coco/instances_val2017.json",
      },
      predict: {
        batch_size: 8,
      },
      architecture: {
        use_bfloat16: true,
      },
      train: {
        checkpoint: {
          path: "gs://xl-ml-test-us-central1/data/pretrain/resnet50-checkpoint-2018-02-07",
          prefix: "resnet50/",
        },
        total_steps: error "Must set `train.total_steps`",
        batch_size: error "Must set `train.batch_size`",
        train_file_pattern: "gs://xl-ml-test-us-central1/data/coco/train*",
      },
    },
    command: [
      "python3",
      "official/vision/detection/main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--strategy_type=tpu",
      "--params_override=%s" % std.manifestYamlDoc(self.paramsOverride)
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--mode=train",
    ],
    paramsOverride+: {
      train+: {
        total_steps: 1000,
      },
    },
  },
  local convergence = mixins.Convergence {
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
        batch_size: 64,
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

  configs: [
    retinanet + functional + v2_8,
    retinanet + functional + v3_8,
    retinanet + convergence + v2_8,
    retinanet + convergence + v3_8,
  ],
}
