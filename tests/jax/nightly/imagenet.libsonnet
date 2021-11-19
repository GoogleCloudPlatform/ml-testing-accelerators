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

local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
{
  local functional = mixins.Functional {
    extraFlags:: '--config.num_epochs=1',
    extraConfig:: 'tpu.py',
  },
  local convergence = mixins.Convergence {
    extraConfig:: 'tpu.py',
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    extraFlags+:: '--config.batch_size=1024'
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v4_32 = {
    accelerator: tpus.v4_32,
    extraFlags+:: '--config.batch_size=2048'
  },
  local imagenet = common.runFlaxNightly {
    modelName:: 'imagenet',
    extraDeps:: 'tensorflow==2.6.2 keras==2.6.0 tensorflow-estimator==2.6.0',
  },

  configs: [
    imagenet + functional + v2_8,
    imagenet + functional + v3_8,
    imagenet + functional + v4_8,
    imagenet + functional + v4_32,
  ],
}
