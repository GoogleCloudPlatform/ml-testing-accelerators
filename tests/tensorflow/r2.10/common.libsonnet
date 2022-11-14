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

local common = import '../common.libsonnet';
local experimental = import '../experimental.libsonnet';
local metrics = import 'templates/metrics.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  ModelGardenTest:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'tf-r2.10.0',
    tpuSettings+: {
      softwareVersion: '2.10.0',
    },
    imageTag: 'r2.10.0',
  },
  // Setting the version for TPU VM.
  tpuVm:: experimental.TensorFlowTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: if config.accelerator.version == 4 && config.accelerator.replicas == 1 then
        'tpu-vm-tf-2.10.0-v4'
      else if config.accelerator.version == 4 && config.accelerator.replicas > 1 then
        'tpu-vm-tf-2.10.0-pod-v4'
      else if config.accelerator.replicas == 1 then
        'tpu-vm-tf-2.10.0'
      else
        'tpu-vm-tf-2.10.0-pod',
    },
  },
  TfVisionTest:: self.ModelGardenTest + common.TfNlpVisionMixin {
    scriptConfig+: {
      runnerPath: 'official/vision/train.py',
    },
  },
  TfNlpTest:: self.ModelGardenTest + common.TfNlpVisionMixin {
    scriptConfig+: {
      runnerPath: 'official/nlp/train.py',
    },
  },
  local functional_schedule = '0 7 * * 6',
  Functional:: mixins.Functional {
    schedule:
      if !(self.accelerator.type == 'tpu') || self.accelerator.name == 'v3-8' || self.accelerator.name == 'v4-8' then
        functional_schedule
      else
        null,
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
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
  // Override default schedule for Functional.
  RunNightly:: {
    schedule: functional_schedule,
  },
  Convergence:: mixins.Convergence {
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
                  // TODO(wcromar): Tighten this restriction
                  std_devs: 2.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
}
