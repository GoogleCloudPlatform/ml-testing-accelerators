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

local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local mnist = common.PyTorchTest {
    imageTag: 'r1.11_3.7',
    modelName: 'mnist-3-7',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_mnist.py',
      '--logdir=$(MODEL_DIR)',
      '--datadir=/datasets/mnist-data',
    ],
  },

  local convergence = common.Convergence {
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Accuracy/test': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 98.0,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },

  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    mnist + v2_8 + convergence + timeouts.Hours(1),
    mnist + v3_8 + convergence + timeouts.Hours(1),
  ],
}
