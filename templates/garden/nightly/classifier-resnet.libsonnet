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
  local resnet = base.GardenTest {
    modelName: "classifier-resnet",
    paramsOverride:: {
      train: {
        epochs: error "Must set `train.epochs`",
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
      "--config_file=official/vision/image_classification/configs/examples/resnet/imagenet/tpu.yaml",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--data_dir=$(IMAGENET_DIR)",
      "--model_type=resnet",
      "--dataset=imagenet",
      "--mode=train_and_eval",
      "--params_override=%s" % std.manifestYamlDoc(self.paramsOverride),
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local functional = mixins.Functional {
    paramsOverride+: {
      train+: {
        epochs: 3, 
      },
    },
  },
  local convergence = mixins.Convergence {
    paramsOverride+: {
      train+: {
        epochs: 90, 
      },
    },
    regressionTestConfig+: {
      metric_success_conditions+: {
        epoch_accuracy_final: {
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

  configs: [
    resnet + v2_8 + functional,
    resnet + v3_8 + functional,
    resnet + v2_8 + convergence,
    resnet + v3_8 + convergence,
  ],
}
