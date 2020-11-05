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

local common = import "../common.libsonnet";
local mixins = import "templates/mixins.libsonnet";

{
  ModelGardenTest:: common.ModelGardenTest {
    frameworkPrefix: "tf-r2.4.0",
    tpuSettings+: {
      softwareVersion: "2.4.0",
    },
    imageTag: "r2.4.0",

    metricCollectionConfig+: {
      metric_to_aggregation_strategies+: {
        "examples_per_second": ["average"],
      },
      use_run_name_prefix: true,
    },
    regressionTestConfig+: {
      metric_success_conditions+: {
        "examples_per_second_average": {
          comparison: "greater_or_equal",
          success_threshold: {
            stddevs_from_mean: 2.0,
          },
        },
      },
    },
  },
  # Running functional tests at 10PM PST daily.
  Functional:: mixins.Functional {
    schedule: "0 6 * * *",
    tpuSettings+: {
      preemptible: false,
    },
  },
  # Running convergence tests at Midnight PST daily.
  Convergence:: mixins.Convergence {
    schedule: "0 8 * * *",
    regressionTestConfig+: {
      metric_success_conditions+: {
        "examples_per_second_average": {
          comparison: "greater_or_equal",
          success_threshold: {
            stddevs_from_mean: 2.0,
          },
        },
      },
    },
  },
}
