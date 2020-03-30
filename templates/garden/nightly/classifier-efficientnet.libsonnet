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
      train_dataset: {
        use_per_replica_batch_size: false,
        batch_size: error "Must set `train_dataset.batch_size`",
      },
      eval_dataset: {
        use_per_replica_batch_size: false,
        batch_size: error "Must set `eval_dataset.batch_size`",
      },
    },
    command: [
      "python3",
      "official/vision/image_classification/classifier_trainer.py",
      "--config_file=official/vision/image_classification/configs/examples/efficientnet/imagenet/efficientnet-b0-tpu.yaml",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--data_dir=gs://imagenet-us-central1/train",
      "--model_type=efficientnet",
      "--dataset=imagenet",
      "--mode=train_and_eval",
      "--data_dir=gs://imagenet-us-central1/train",
      "--params_override=%s" % std.manifestYamlDoc(self.paramsOverride)
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
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    paramsOverride+: {
      train_dataset+: {
        batch_size: 1024,
      },
      eval_dataset+: {
        batch_size: 1024,
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    paramsOverride+: {
      train_dataset+: {
        batch_size: 1024,
      },
      eval_dataset+: {
        batch_size: 1024,
      },
    },
  },

  configs: [
    efficientnet + v2_8 + functional,
    efficientnet + v3_8 + functional,
    efficientnet + v2_8 + convergence,
    efficientnet + v3_8 + convergence,
  ],
}
