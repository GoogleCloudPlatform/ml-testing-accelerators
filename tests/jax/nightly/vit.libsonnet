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

local common = import '../common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
{
  local functional = mixins.Functional {
    extraFlags:: '--config.total_steps=1000',
  },
  local convergence = mixins.Convergence {
  },

  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },

  // TODO: move shared code into a common file
  local vit = common.JaxTest + common.jaxlibHead + common.nightlyImage {
    local config = self,
    frameworkPrefix: 'flax-nightly',
    modelName:: 'vit',
    extraDeps:: '',
    extraFlags:: '',
    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      pip install --upgrade pip
      pip install --upgrade clu %(extraDeps)s

      echo "Checking out and installing JAX..."
      git clone https://github.com/google/jax.git
      cd jax
      echo "jax git hash: $(git rev-parse HEAD)"
      %(installLocalJax)s
      %(maybeBuildJaxlib)s
      %(printDiagnostics)s

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      if [ "$num_devices" = "1" ]; then
        echo "No TPU devices detected"
        exit 1
      fi

      git clone https://github.com/google-research/vision_transformer.git
      cd vision_transformer
      pip install -r vit_jax/requirements.txt

      python3 -m vit_jax.main --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 --workdir=$(MODEL_DIR) \
        --config.pretrained_dir="gs://vit_models/imagenet21k" %(extraFlags)s
    ||| % (self.scriptConfig {
             extraFlags: config.extraFlags,
             extraDeps: config.extraDeps,
           }),
  },

  configs: [
    vit + functional + v2_8,
    vit + convergence + v3_8,
  ],
}
