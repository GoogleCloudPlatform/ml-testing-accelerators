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

local base = import "../base.libsonnet";

{
  local PyTorchBaseTest = base.BaseTest {
    regressionTestConfig+: {
      metric_subset_to_alert: [
        "ExecuteTime__Percentile_99_sec_final",
        "CompileTime__Percentile_99_sec_final",
        "total_wall_time",
        "Accuracy/test_final",
        "aten_ops_sum_final",
      ],
      metric_success_conditions+: {
        "ExecuteTime__Percentile_99_sec_final": {
          success_threshold: {
            stddevs_from_mean: 5.0,
          },
          comparison: "less",
          wait_for_n_points_of_history: 10,
        },
        "CompileTime__Percentile_99_sec_final": {
          success_threshold: {
            stddevs_from_mean: 5.0,
          },
          comparison: "less",
          wait_for_n_points_of_history: 10,
        },
        "aten_ops_sum_final": {
          success_threshold: {
            stddevs_from_mean: 0.0,
          },
          comparison: "less_or_equal",
        },
      },
    },

    metricCollectionConfig+: {
      "tags_to_ignore": ["LearningRate"]
    },
  },
  PyTorchTest:: PyTorchBaseTest {
    image: "gcr.io/xl-ml-test/pytorch-xla",

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              envMap+: {
                XLA_USE_BF16: "0",
              },
              resources+: {
                requests+: {
                  cpu: "4.5",
                  memory: "8Gi",
                },
              },

              volumeMounts: [{
                mountPath: "/dev/shm",
                name: "dshm",
              }],
            },
          },
          volumes: [
            {
              name: "dshm",
              emptyDir: {
                medium: "Memory",
              },
            },
          ],
        },
      },
    },
  },
  # Pod tests are run by creating an instance group to feed the TPU Pods
  PyTorchPodTest:: PyTorchBaseTest {
    local config = self,

    image: "gcr.io/xl-ml-test/pytorch-pods",
    instanceType: "n1-standard-8",
    condaEnv: "torch-xla-nightly",
    xlaDistFlags: "",

    jobSpec+:: {
      backoffLimit: 0,
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              envMap+: {
                MACHINE_TYPE: config.instanceType,
                ACCELERATOR_TYPE: config.acceleratorName,
                CONDA_ENV: config.condaEnv,
                XLA_DIST_FLAGS: config.xlaDistFlags,
              },
            },
          },
        },
      },
    },
  },
}
