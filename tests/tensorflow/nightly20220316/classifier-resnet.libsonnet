// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local gpus = import 'templates/gpus.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local resnet = common.ModelGardenTest {
    modelName: 'classifier-resnet',
    paramsOverride:: {
      train: {
        epochs: error 'Must set `train.epochs`',
      },
      evaluation: {
        epochs_between_evals: error 'Must set `evaluation.epochs_between_evals`',
      },
      train_dataset: {
        builder: 'records',
      },
      validation_dataset: {
        builder: 'records',
      },
    },
    command: [
      'python3',
      'official/legacy/image_classification/classifier_trainer.py',
      '--data_dir=$(IMAGENET_DIR)',
      '--model_type=resnet',
      '--dataset=imagenet',
      '--mode=train_and_eval',
      '--model_dir=$(MODEL_DIR)',
      '--params_override=%s\n' % std.manifestYamlDoc(self.paramsOverride),
    ],
  },
  local functional = common.Functional {
    paramsOverride+: {
      train+: {
        epochs: 1,
      },
      evaluation+: {
        epochs_between_evals: 1,
      },
    },
  },
  local convergence = common.Convergence {
    paramsOverride+: {
      train+: {
        epochs: 90,
      },
      evaluation+: {
        epochs_between_evals: 1,
      },
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'validation/epoch_accuracy': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.76,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
                  // TODO(wcromar): Tighten this restriction
                  std_devs: 4.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
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
      '--config_file=official/legacy/image_classification/configs/examples/resnet/imagenet/gpu.yaml',
    ],
  },
  local k80 = gpu_common {
    local config = self,

    paramsOverride+:: {
      train_dataset+: {
        batch_size: 128,
      },
      validation_dataset+: {
        batch_size: 128,
      },
    },
    accelerator: gpus.teslaK80,
  },
  local k80x8 = k80 {
    paramsOverride+:: {
      runtime+: {
        all_reduce_alg: 'hierarchical_copy',
      },
    },
    accelerator: gpus.teslaK80 { count: 8 },
  },
  local v100 = gpu_common {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
  },
  local v100x8 = v100 {
    accelerator: gpus.teslaV100 { count: 8 },
  },
  local a100x4 = gpu_common {
    paramsOverride+:: {
      train_dataset+: {
        batch_size: 512,
      },
      validation_dataset+: {
        batch_size: 512,
      },
    },

    accelerator: gpus.teslaA100 { count: 4 },
  },

  local tpu_common = {
    command+: [
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--config_file=official/legacy/image_classification/configs/examples/resnet/imagenet/tpu.yaml',
    ],
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
  local v4_8 = tpu_common {
    accelerator: tpus.v4_8,
  },
  local v4_32 = tpu_common {
    accelerator: tpus.v4_32,
  },
  local tpuVm = experimental.TensorFlowTpuVmMixin,

  configs: [
    resnet + v100x8 + functional + mixins.Unsuspended,
    resnet + v100x8 + convergence + timeouts.Hours(14),
    resnet + a100x4 + convergence,
    resnet + v2_8 + functional,
    resnet + v2_8 + functional + tpuVm,
    resnet + v3_8 + functional,
    resnet + v3_8 + functional + tpuVm,
    resnet + v2_8 + convergence,
    resnet + v3_8 + convergence,
    resnet + v2_32 + functional + common.RunNightly,
    resnet + v2_32 + functional + tpuVm + common.RunNightly,
    resnet + v3_32 + functional,
    resnet + v3_32 + functional + tpuVm,
    resnet + v3_32 + convergence,
    resnet + v4_8 + functional + tpuVm,
    resnet + v4_32 + functional + tpuVm,
  ],
}
