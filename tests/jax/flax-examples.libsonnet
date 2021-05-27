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
  local runFlaxExample =
    common.JaxTest + common.Convergence + common.jaxlibHead + common.libtpuNightly + tpus.v2_8 {
      local config = self,

      frameworkPrefix: 'flax',
      extraDeps:: '',
      extraFlags:: '',

      testScript:: |||
        set -x
        set -u
        set -e

        # .bash_logout sometimes causes a spurious bad exit code, remove it.
        rm .bash_logout

        pip install --upgrade pip

        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git
        cd jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -e .
        %(installJaxlib)s

        pip install --upgrade clu %(extraDeps)s

        num_devices=`python3 -c "import jax; print(jax.device_count())"`
        if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
        fi

        git clone https://github.com/google/flax
        cd flax
        pip install -e .
        cd examples/%(modelName)s

        export GCS_BUCKET=$(MODEL_DIR)
        export TFDS_DATA_DIR=$(TFDS_DIR)

        python3 main.py --workdir=$(MODEL_DIR)  --config=configs/default.py %(extraFlags)s
      ||| % (self.scriptConfig {
               modelName: config.modelName,
               extraDeps: config.extraDeps,
               extraFlags: config.extraFlags,
             }),
    },
  local functional = common.Functional {
    extraFlags:: '--config.num_epochs=1',
  },
  local convergence = common.Convergence {
    extraFlags:: '--config.num_epochs=10',
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local imagenet = runFlaxExample {
    modelName:: 'imagenet',
    //extraFlags:: '--config.num_epochs=2',
  },

  local wmt = runFlaxExample {
    modelName:: 'wmt',
    extraDeps:: 'tensorflow_text sentencepiece',
    extraFlags:: '--config.num_train_steps=10 --config.per_device_batch_size=16',
  },
  local mnist = runFlaxExample {
    modelName:: 'mnist',
  },

  configs: [
    imagenet + functional + v2_8,
    wmt + v2_8,
    mnist + v3_8 + convergence,
    mnist + v2_8 + functional,
    mnist + v2_8 + convergence,
  ],
}
