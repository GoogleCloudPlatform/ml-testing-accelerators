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
  local syncDevicesTest = common.JaxTest + mixins.Functional {
    modelName: '%s-jax-orbax-async' % [self.jaxlibVersion],

    // Trigger the test at 13:00 UTC.
    schedule: '0 13 * * *',
    tpuSettings+: {
      slices: 2,
    },
    testScript:: |||
      set -x
      set -u
      set -e
      . ~/.profile
      %(printDiagnostics)s
      export TPU_NAME=local
      export TPU_STDERR_LOG_LEVEL=0
      export TPU_MIN_LOG_LEVEL=0
      export JAX_USE_PJRT_C_API_ON_TPU=1
      export TF_CPP_MIN_LOG_LEVEL=0
      export MP_MODEL_DIR=gs://multipod-xlml-outputs${MODEL_DIR:4:500}
      git clone --single-branch --branch multipod-tests https://github.com/GoogleCloudPlatform/ml-testing-accelerators.git --depth=1
      pip3 install google-cloud-storage
      pip3 install importlib-resources
      pip3 install etils
      pip3 install orbax
      pip3 install portpicker
      pip3 install tensorflow
      python3 ml-testing-accelerators/tests/multipods/jax/unit_tests/orbax_async_checkpoint_test.py --bucket_path=gs://multipod-xlml-outputs${MODEL_DIR:4:500} --ckpt_dir=orbax-checkpoints
      exit 0
    ||| % self.scriptConfig,
  },
  local v4_16 = {
    accelerator: tpus.v4_16,
  },
  configs: [
    syncDevicesTest + common.jaxlibNightly + common.tpuVmV4Base + v4_16,
  ],
}
