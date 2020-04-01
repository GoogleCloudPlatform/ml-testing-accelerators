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
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local resnet50_MP = base.PyTorchTest {
    modelName: "resnet50-MP",
    command: [
      "python3",
      "pytorch/xla/test/test_train_mp_imagenet.py",
      "--logdir=$(MODEL_DIR)",
      "--model=resnet50",
      "--num_workers=8",
      "--batch_size=128",
      "--log_steps=200",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          containers: [
            container {
              resources+: {
                requests: {
                  cpu: "90.0",
                  memory: "400Gi",
                },
              },
            } for container in super.containers
          ],
        },
      },
    },
  },
  local functional = base.Functional {
    command+: [
      "--num_epochs=2",
      "--datadir=/datasets/imagenet-mini",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "imagenet-mini-pd",
              gcePersistentDisk: {
                pdName: "imagenet-mini-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "imagenet-mini-pd",
                readOnly: true,
              }],
            } for container in super.containers
          ],
        },
      },
    },
  },
  local convergence = base.Convergence {
    accelerator+: tpus.Preemptible,
    command+: [
      "--num_epochs=90",
      "--datadir=/datasets/imagenet",
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 75.0,
          },
          comparison: "greater",
        },
      },
    },
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "imagenet-pd",
              gcePersistentDisk: {
                pdName: "imagenet-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "imagenet-pd",
                readOnly: true,
              }],
            } for container in super.containers
          ],
        },
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    resnet50_MP + v3_8 + convergence + timeouts.Hours(26),
    resnet50_MP + v3_8 + functional + timeouts.Hours(2),
  ],
}
