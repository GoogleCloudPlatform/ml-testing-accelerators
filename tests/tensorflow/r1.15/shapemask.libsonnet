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
  local shapemask = base.LegacyTpuTest {
    modelName: "shapemask",
    paramsOverride: {
      eval: {
        eval_file_pattern: '$(COCO_DIR)/val*',
        eval_samples: 5000,
        eval_batch_size: 8,
        val_json_file: '$(COCO_DIR)/instances_val2017.json',
      },
      predict: {
        predict_batch_size: 8,
      },
      train: {
        iterations_per_loop: error "Must set `iterations_per_loop`",
        train_batch_size: error "Must set `train_batch_size`",
        checkpoint: {
          path: '$(RESNET_PRETRAIN_DIR)/resnet50-checkpoint-2018-02-07',
          prefix: 'resnet50/',
        },
        total_steps: 22500,
        train_file_pattern: '$(COCO_DIR)/train*',
      },
      shapemask_head: {
        use_category_for_mask: true,
        shape_prior_path: '$(SHAPEMASK_DIR)/kmeans_class_priors_91x20x32x32.npy',
      },
    },
    command: [
      "python3",
      "/shapemask/models/official/detection/main.py",
      "--model=shapemask",
      "--use_tpu=True",
      "--eval_after_training=False",
      "--mode=train",
      "--params_override=%s" % (std.manifestYamlDoc(self.paramsOverride) + "\n"),
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--model_dir=$(MODEL_DIR)",
    ],

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              env+: [
                {
                  name: "PYTHONPATH",
                  value: "/shapemask/models/",
                },
              ],
            },
          },
        },
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train+: {
        iterations_per_loop: 100,
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
        iterations_per_loop: 100,
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
        iterations_per_loop: 5000,
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
        iterations_per_loop: 5000,
        train_batch_size: 256,
      },
    },
    command+: [
      "--num_cores=32",
    ],
  },
  local convergence = base.Convergence,

  configs: [
    shapemask + v2_8 + convergence,
    shapemask + v3_8 + convergence,
    shapemask + v2_32 + convergence,
    shapemask + v3_32 + convergence,
  ],
}
