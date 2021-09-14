// Copyright 2021 Google LLC
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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local operations = common.PyTorchTest {
    modelName: 'python-ops',
    command: [
      'bash',
      'pytorch/xla/test/run_tests.sh',
    ],
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap:: {},
        },
        literals: {},
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local py_ops_tpu_vm = common.PyTorchTest {
    modelName: 'python-ops',
    schedule: '22 17 * * *',
    command: utils.scriptCommand(
      |||
        %(command_common)s
        sudo pip3 install hypothesis
        cd xla/test
        export TPUVM_MODE=1
        ./run_tests.sh
      ||| % common.tpu_vm_1_9_install
    ),
  },
  configs: [
    operations + v2_8 + common.Functional + timeouts.Hours(2),
    operations + v3_8 + common.Functional + timeouts.Hours(2),
    py_ops_tpu_vm + v3_8 + common.Functional + timeouts.Hours(2) + experimental.PyTorchTpuVmMixin,
    py_ops_tpu_vm + v2_8 + common.Functional + timeouts.Hours(2) + experimental.PyTorchTpuVmMixin,
  ],
}
