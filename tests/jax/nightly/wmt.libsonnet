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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
{
  local functional = mixins.Functional {
    extraFlags:: '--config.num_train_steps=10 --config.per_device_batch_size=16',
    extraConfig:: 'default.py',
  },
  local convergence = mixins.Convergence {
    extraConfig:: 'default.py',
    extraFlags:: '--config.reverse_translation=True  --config.per_device_batch_size=32',
  },
  local v4_32 = {
    accelerator: tpus.v4_32,
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local wmt = common.runFlaxNightly {
    modelName:: 'wmt',
    extraDeps:: 'tensorflow_text sentencepiece tensorflow==2.6.2 keras==2.6.0 tensorflow-estimator==2.6.0',
  },
  configs: [
    wmt + functional + v2_8,
    wmt + functional + v4_8,
    wmt + functional + v4_32,
    wmt + convergence + v3_8 + timeouts.Hours(20),
  ],
}
