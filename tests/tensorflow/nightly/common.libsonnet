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
local metrics = import 'templates/metrics.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  ModelGardenTest:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'tf-nightly',
    tpuSettings+: {
      softwareVersion: 'nightly',
    },
    imageTag: 'nightly',

    metricConfig: metrics.MetricCollectionConfigHelper {
      sourceMap:: {
        tensorboard: metrics.TensorBoardSourceHelper {
          exclude_tags: [

          ],
          include_tags: [
            {
              strategies: [
                'FINAL',
              ],
              tag_pattern: '*',
            },
          ],
          merge_runs: false,
        },
        literals: {
          assertions: {
            duration: {
              inclusive_bounds: false,
              std_devs_from_mean: {
                comparison: 'LESS',
                std_devs: 5,
              },
              wait_for_n_data_points: 10,
            },
          },
        },
      },
    },
  },
  Functional:: mixins.Functional + mixins.Suspended {
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
  ServingTest:: common.ServingTest {
    local config = self,
    image: 'gcr.io/xl-ml-test/allencwang-load-test',
    frameworkPrefix: 'tf-nightly',
    servingConfig+: {
      modelServerImage: 'gcr.io/xl-ml-test/allencwang-tf-serving-tpu:latest',
    },
  },
}
