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
  local resnet = common.ModelGardenTest {
    modelName: "classifier-resnet",
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
      "--data_dir=$(IMAGENET_DIR)",
      "--model_type=resnet",
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
        epochs: 90, 
      },
      evaluation+: {
        epochs_between_evals: 1,
      },
    },
    regressionTestConfig+: {
      metric_success_conditions+: {
        "validation/epoch_accuracy_final": {
          success_threshold: {
            fixed_value: 0.76,
          },
          comparison: "greater",
        },
      },
    },
  },

  local gpu_common = {
    local config = self,

    paramsOverride+:: {
      runtime+: {
        num_gpus: config.accelerator.count,
      },
    },
    command+: [
      "--config_file=official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml",
    ],
  },
  local v100 = gpu_common {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = gpu_common {
    accelerator: gpus.teslaV100 + { count: 4 },
  },
  local v100x8 = gpu_common {
    accelerator: gpus.teslaV100 + { count: 8 },
  },

  local k80 = gpu_common {
    paramsOverride+:: {
      train_dataset+: {
        batch_size: 128
      },
      validation_dataset+: {
        batch_size: 128
      },
    },
    accelerator: gpus.teslaK80,
  },

  local tpu_common = {
    command+: [
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--config_file=official/vision/image_classification/configs/examples/resnet/imagenet/tpu.yaml",
    ]
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
  },
  local v3_8 = tpu_common {
    accelerator: tpus.v3_8,
  },
  local v2_32 = tpu_common {
    accelerator: tpus.v2_32,
  },
  local v3_32 = tpu_common {
    accelerator: tpus.v3_32,
  },

  configs: [
    resnet + v100 + functional + timeouts.Hours(3),
    resnet + v100x4 + functional + mixins.Experimental,
    resnet + v100x4 + convergence + mixins.Experimental,
    resnet + v100x8 + functional + mixins.Experimental,
    resnet + k80 + functional + timeouts.Hours(5) + mixins.Experimental,
    resnet + k80 + convergence + timeouts.Hours(48) + mixins.Experimental,
    resnet + v2_8 + functional,
    resnet + v3_8 + functional,
    resnet + v2_8 + convergence,
    resnet + v3_8 + convergence,
    resnet + v2_32 + functional,
    resnet + v3_32 + functional,
    resnet + v2_32 + convergence,
    resnet + v3_32 + convergence,
  ],
}
