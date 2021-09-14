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

    frameworkPrefix: 'tf-nightly',
    tpuSettings+: {
      softwareVersion: 'nightly',
    },
    imageTag: 'nightly',

    metricCollectionConfig+: {
      metric_to_aggregation_strategies+: {
        examples_per_second: ['average'],
      },
      use_run_name_prefix: true,
    },
    regressionTestConfig+: {
      metric_success_conditions+: {
        examples_per_second_average: {
          comparison: 'greater_or_equal',
          success_threshold: {
            stddevs_from_mean: 2.0,
          },
        },
      },
    },
  },
  local functional_schedule = '0 7 * * *',
  Functional:: mixins.Functional {
    // Only schedule v3-8 TPU tests by default
    schedule:
      if !(self.accelerator.type == 'tpu') || self.accelerator.name == 'v3-8' then
        functional_schedule
      else
        null,
    regressionTestConfig+: {
      metric_success_conditions+: {
        examples_per_second_average: {
          comparison: 'greater_or_equal',
          success_threshold: {
            stddevs_from_mean: 4.0,
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
    regressionTestConfig+: {
      metric_success_conditions+: {
        examples_per_second_average: {
          comparison: 'greater_or_equal',
          success_threshold: {
            stddevs_from_mean: 2.0,
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
