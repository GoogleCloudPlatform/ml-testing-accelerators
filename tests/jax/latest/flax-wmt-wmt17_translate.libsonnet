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
  local profile = self.profile,
  profile:: mixins.Functional {
    mode: 'profile',
    extraFlags+:: ['--config.num_train_steps=40', '--config.per_device_batch_size=16'],
    extraDeps+:: ['protobuf==3.20.*'],
    extraConfig:: 'default.py',
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  local wmt = self.wmt,
  wmt:: common.runFlaxLatest {
    folderName:: 'wmt',
    modelName:: 'wmt-wmt17.translate',
    extraDeps+:: ['tf-nightly-cpu tensorflow-datasets tensorflow-text-nightly sentencepiece'],
  },
  local wmt_profiling = self.wmt_profiling,
  wmt_profiling:: wmt {
    local config = self,
    runTest+: |||
      gsutil -q stat $(MODEL_DIR)/plugins/profile/*/*.xplane.pb
      gsutil cp -r $(MODEL_DIR)/plugins /tmp/
      python3 -m pip uninstall tensorboard_plugin_profile
      python3 -m pip install tbp-nightly
      python3 ~/.local/lib/python3.*/site-packages/tensorboard_plugin_profile/integration_tests/tpu/tensorflow/tpu_tf2_keras_test.* --log_directory=/tmp/
    ||| % (self.scriptConfig),
  },
  configs: [
    wmt_profiling + profile + v3_8 + timeouts.Hours(1),
  ],
}
