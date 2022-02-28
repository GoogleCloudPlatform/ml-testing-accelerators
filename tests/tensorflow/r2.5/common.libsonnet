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
local mixins = import 'templates/mixins.libsonnet';

{
  ModelGardenTest:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'tf-r2.5.0',
    tpuSettings+: {
      softwareVersion: '2.5.0',
    },
    imageTag: 'r2.5.0',
  },
  // Don't run tests by default since the release is stable.
  Functional:: mixins.Functional + mixins.Suspended {
    schedule: null,
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
  // Don't run tests by default since the release is stable.
  Convergence:: mixins.Convergence {
    schedule: null,
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
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
