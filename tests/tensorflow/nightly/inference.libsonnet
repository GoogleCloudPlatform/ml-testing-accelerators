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

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local gpus = import 'templates/gpus.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local inference = common.ModelGardenTest {
    modelName: 'inference',
    command: [
      'curl',
      '-v',
      '-d',
      '\'{"instances": [1.0, 2.0, 5.0]}\'',
      'http://$(cat /scripts/tpu_ip):8501/v1/models/half_plus_two:predict',
    ],
  },
  local functional = common.Functional,
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  configs: [
    inference + functional + v2_8 + experimental.TensorflowServingTpuVmMixin + mixins.Experimental,
  ],
}
