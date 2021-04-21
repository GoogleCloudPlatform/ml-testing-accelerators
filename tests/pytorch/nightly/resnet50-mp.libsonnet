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

local common = import "common.libsonnet";
local experimental = import "tests/experimental.libsonnet";
local gpus = import "templates/gpus.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local mixins = import "templates/mixins.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local gpu_command_base = |||
    unset XRT_TPU_CONFIG
    export GPU_NUM_DEVICES=%(num_gpus)s
    python3 pytorch/xla/test/test_train_mp_imagenet.py \
    --model=resnet50 \
    --batch_size=128 \
    --log_steps=100 \
    --num_workers=4 \
    --num_epochs=2 \
    --datadir=/datasets/imagenet-mini \
  |||,
  local resnet50_gpu = common.PyTorchTest {
    imageTag: 'nightly_3.6_cuda',
    modelName: 'resnet50-mp',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '13.0',
    memory: '40Gi',
  },

  local resnet50_MP = common.PyTorchTest {
    modelName: 'resnet50-mp',
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_imagenet.py',
      '--logdir=$(MODEL_DIR)',
      '--model=resnet50',
      '--num_workers=8',
      '--batch_size=128',
      '--log_steps=200',
    ],
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: "90.0",
                memory: "400Gi",
              },
            },
          },
        },
      },
    },
  },

  local functional = common.Functional {
    command+: [
      '--num_epochs=2',
      '--datadir=/datasets/imagenet-mini',
    ],
  },
  local functional_fake_data = common.Functional {
    command+: [
      "--num_epochs=2",
      "--fake_data",
    ],
    volumeMap+: {
      datasets: null,
    },
  },
  local convergence = common.Convergence {
    command+: [
      '--num_epochs=90',
      '--datadir=/datasets/imagenet',
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        'Accuracy/test_final': {
          success_threshold: {
            fixed_value: 75.0,
          },
          comparison: 'greater',
        },
      },
    },
  },
  local functional = common.Functional {
    command+: [
      "--num_epochs=2",
      "--datadir=/datasets/imagenet-mini",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  local v100 = {
    accelerator: gpus.teslaV100,
    command: utils.scriptCommand(
      gpu_command_base % 1
    ),
    schedule: '0 20 * * *',
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 { count: 4 },
    command: utils.scriptCommand(
      gpu_command_base % 4
    ),
    schedule: '2 20 * * *',
  },
  local resnet50_tpu_vm = experimental.PyTorchTpuVmTest {
    # This test uses the default pytorch XLA version built into the TPUVM, which
    # is 1.8.1 as of Apr 19.
    frameworkPrefix: "pt-r1.8.1",
    modelName: "resnet50-mp",
    paramsOverride: {
      num_epochs: error "Must set `num_epochs`",
      datadir: error "Must set `datadir`",
    },
    command: utils.scriptCommand(
      |||
        git clone https://github.com/pytorch/xla.git -b r1.8.1
        pip3 install tensorboardX google-cloud-storage
        python3 xla/test/test_train_mp_imagenet.py \
          --logdir=$(MODEL_DIR) \
          --datadir=%(datadir)s \
          --model=resnet50 \
          --num_workers=8 \
          --batch_size=128 \
          --log_steps=200 \
          --num_epochs=%(num_epochs)d \
      ||| % self.paramsOverride,
    ),
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: "1",
                memory: "2Gi",
              },
            },
          },
        },
      },
    },
  },
  local functional_tpu_vm = common.Functional {
    paramsOverride: {
      num_epochs: 2,
      datadir: "/datasets/imagenet-mini",
    },
  },
  local convergence_tpu_vm = common.Convergence {
    paramsOverride: {
      num_epochs: 5,
      datadir: "/datasets/imagenet",
    },
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 30.0,
          },
          comparison: "greater",
        },
      },
    },
  },
  configs: [
    resnet50_MP + v3_8 + convergence + timeouts.Hours(26) + mixins.PreemptibleTpu,
    resnet50_MP + v3_8 + functional + timeouts.Hours(2),
    common.PyTorchTest + resnet50_tpu_vm + v3_8 + functional_tpu_vm + timeouts.Hours(2),
    common.PyTorchTest + resnet50_tpu_vm + v3_8 + convergence_tpu_vm + timeouts.Hours(4),
    resnet50_gpu + common.Functional + v100 + timeouts.Hours(2),
    resnet50_gpu + common.Functional + v100x4 + timeouts.Hours(1),
  ],
}
