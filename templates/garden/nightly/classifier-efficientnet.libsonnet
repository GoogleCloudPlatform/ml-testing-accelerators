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
  local efficientnet = base.GardenTest {
    modelName: "classifier-efficientnet",
    paramsOverride:: {
      train: {
        epochs: error "Must set `train.epochs`",
      },
      evaluation: {
        epochs_between_evals: error "Must set `evaluation.epochs_between_evals`",
      },
      train_dataset: {
        builder: "records",
      },
      validation_dataset: {
        builder: "records",
      },
    },
    command: [
      "python3",
      "official/vision/image_classification/classifier_trainer.py",
      "--config_file=official/vision/image_classification/configs/examples/efficientnet/imagenet/efficientnet-b0-tpu.yaml",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--data_dir=$(IMAGENET_DIR)",
      "--model_type=efficientnet",
      "--dataset=imagenet",
      "--mode=train_and_eval",
      "--model_dir=$(MODEL_DIR)",
      "--params_override=%s" % std.manifestYamlDoc(self.paramsOverride),
    ],
  },
  local functional = mixins.Functional {
    paramsOverride+: {
      train+: {
        epochs: 3, 
      },
      evaluation+: {
        epochs_between_evals: 3,
      },
    },
  },
  local convergence = mixins.Convergence {
    paramsOverride+: {
      train+: {
        epochs: 500, 
      },
      evaluation+: {
        epochs_between_evals: 500,
      },
    },
    regressionTestConfig+: {
      metric_success_conditions+: {
        val_epoch_accuracy_final: {
          success_threshold: {
            fixed_value: 0.76,
          },
          comparison: "greater",
        },
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },

  configs: [
    efficientnet + v2_8 + functional,
    efficientnet + v3_8 + functional,
    efficientnet + v2_8 + convergence + timeouts.Hours(31),
    efficientnet + v3_8 + convergence + timeouts.Hours(31),
    efficientnet + v2_32 + convergence + timeouts.Hours(30),
    efficientnet + v3_32 + convergence + timeouts.Hours(24),
  ],
}
