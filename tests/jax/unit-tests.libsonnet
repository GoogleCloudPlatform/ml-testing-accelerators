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
  local runUnitTests = common.JaxTest + mixins.Functional {
    modelName: '%s-libtpu-%s' % [self.jaxlibVersion, self.libtpuVersion],

    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      pip install --upgrade pip
      pip install --upgrade numpy==1.18.5 scipy wheel future six cython pytest \
          absl-py opt-einsum msgpack

      echo "Checking out and installing JAX..."
      git clone https://github.com/google/jax.git
      cd jax
      echo "jax git hash: $(git rev-parse HEAD)"
      pip install -e .

      %(installJaxlib)s

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      if [ "$num_devices" = "1" ]; then
        echo "No TPU devices detected"
        exit 1
      fi

      export JAX_NUM_GENERATED_CASES=5
      export COLUMNS=160
      # Remove 'Captured stdout call' due to b/181896778
      python3 -u -m pytest --tb=short tests examples | sed 's/Captured stdout call/output/'
      exit ${PIPESTATUS[0]}
    ||| % self.scriptConfig,
  },

  configs: [
    runUnitTests + common.jaxlibHead + common.libtpuNightly,
    runUnitTests + common.jaxlibLatest + common.libtpuNightly,
    runUnitTests + common.jaxlibHead + common.libtpuAlpha,
    runUnitTests + common.jaxlibLatest + common.libtpuAlpha,
  ],
}
