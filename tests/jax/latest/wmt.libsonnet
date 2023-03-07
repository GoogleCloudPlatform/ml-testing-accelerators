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
    extraFlags+:: ['--config.num_train_steps=10', '--config.per_device_batch_size=16'],
    extraConfig:: 'default.py',
  },
  local convergence = mixins.Convergence {
    extraConfig:: 'default.py',
    extraFlags+:: ['--config.reverse_translation=True', '--config.per_device_batch_size=32', '--config.num_train_steps=70000'],
  },
  local profile = mixins.Functional {
    mode: 'profile',
    extraFlags+:: ['--config.num_train_steps=40', '--config.per_device_batch_size=16'],
    extraConfig:: 'default.py',
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local wmt = common.runFlaxLatest {
    modelName:: 'wmt',
    extraDeps+:: ['tensorflow-cpu tensorflow-datasets tensorflow_text sentencepiece'],
  },
  local wmt_profiling = wmt {
    local config = self,
    testScript+:: |||
      gsutil -q stat $(MODEL_DIR)/plugins/profile/*/*.xplane.pb
    ||| % (self.scriptConfig {}),
  },
  configs: [
    wmt + functional + v2_8,
    wmt + convergence + v3_32,
    wmt_profiling + profile + v3_8 + timeouts.Minutes(40),
  ],
}
