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
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local resnet = common.ModelGardenTest {
    modelName: 'resnet-ctl',
    command: [
      'python3',
      'official/legacy/image_classification/resnet/resnet_ctl_imagenet_main.py',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--distribution_strategy=tpu',
      '--use_synthetic_data=false',
      '--dtype=fp32',
      '--enable_eager=true',
      '--enable_tensorboard=true',
      '--log_steps=50',
      '--single_l2_loss_op=true',
      '--use_tf_function=true',
      '--data_dir=$(IMAGENET_DIR)',
      '--model_dir=$(MODEL_DIR)',
    ],
    tpuSettings+: {
      softwareVersion: 'test_nightly_direct_path_non_distrib_cache',
    },
  },
  local functional = common.Functional {
    command+: [
      '--train_epochs=1',
      '--epochs_between_evals=1',
    ],
  },
  local convergence = common.Convergence {
    command+: [
      '--train_epochs=90',
      '--epochs_between_evals=90',
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: ['--batch_size=1024'],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: ['--batch_size=2048'],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: ['--batch_size=4096'],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: ['--batch_size=8192'],
  },
  local tpuVm = experimental.TensorFlowTpuVmMixin,

  configs: [
    resnet + v2_8 + functional,
    resnet + v2_8 + functional + tpuVm,
    resnet + v3_8 + functional,
    resnet + v2_8 + convergence + timeouts.Hours(16),
    resnet + v3_8 + convergence,
    resnet + v2_32 + functional + tpuVm,
    resnet + v3_32 + functional,
    resnet + v3_32 + convergence,
  ],
}
