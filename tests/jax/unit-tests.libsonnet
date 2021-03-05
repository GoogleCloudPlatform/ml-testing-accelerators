# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

local common = import "common.libsonnet";
local mixins = import "templates/mixins.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local utils = import 'templates/utils.libsonnet';

{
  local runUnitTests = common.JaxTest + mixins.Functional {
    jaxlibVersion:: error "Add jaxlib version mixin",
    libtpuVersion:: error "Include libtpu version mixin",
    scriptConfig:: {
      installJaxlib: error "Must define `installJaxlib`",
    },

    modelName: "%s-libtpu-%s" % [self.jaxlibVersion, self.libtpuVersion],

    testScript:: |||
      set -x
      set -u
      set -e
      pip install --upgrade pip
      pip install --upgrade numpy==1.18.5 scipy wheel future six cython pytest absl-py opt-einsum msgpack

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

    accelerator: tpus.v2_8,
  },

  local jaxlibHead = {
    jaxlibVersion:: "head",
    scriptConfig:: {
      installJaxlib: |||
        echo "Building jaxlib from source at TF head"
        echo "Checking out TF..."
        cd ~/
        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        echo "TensorFlow git hash: $(git rev-parse HEAD)"

        echo "Building JAX..."
        cd ~/jax
        python3 build/build.py --bazel_options="--override_repository=org_tensorflow=$HOME/tensorflow"
        pip install dist/*.whl
      |||,
    },
  },

  local jaxlibLatest = {
    jaxlibVersion:: "latest",
    scriptConfig:: {
      installJaxlib: |||
        pip install --upgrade jaxlib
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
      |||,
    },
  },

  local libtpuNightly = {
    libtpuVersion: "nightly",
    tpuSettings+: {
      softwareVersion: "v2-nightly",
    },
  },

  local libtpuAlpha = {
    libtpuVersion: "alpha",
    tpuSettings+: {
      softwareVersion: "v2-alpha",
    },
  },

  configs: [
    runUnitTests + jaxlibHead + libtpuNightly,
    runUnitTests + jaxlibLatest + libtpuNightly,
    runUnitTests + jaxlibHead + libtpuAlpha,
    runUnitTests + jaxlibLatest + libtpuAlpha,
  ]
}
