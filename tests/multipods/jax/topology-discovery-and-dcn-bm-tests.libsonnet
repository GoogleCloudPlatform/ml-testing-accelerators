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

{
  local runTopologyDiscoveryAndDcnBmTest = common.JaxTest + mixins.Functional {
    modelName: '%s-topology-discovery' % [self.jaxlibVersion],

    testScript:: |||
      set +x
      set -u
      set -e

      . ~/.profile

      %(printDiagnostics)s

      export TPU_NAME=local
      export TPU_STDERR_LOG_LEVEL=0
      export TPU_MIN_LOG_LEVEL=0
      export TPU_VMODULE=tpu_configuration_ops_impl=3
      export JAX_USE_PJRT_C_API_ON_TPU=1
      export TF_CPP_MIN_LOG_LEVEL=0

      python3 -c "import jax; print(jax.numpy.add(1,1))"

      exit 0
    ||| % self.scriptConfig,
  },

  configs: [
    runTopologyDiscoveryAndDcnBmTest + common.jaxlibHead + common.tpuVmV4Base,
  ],
}
