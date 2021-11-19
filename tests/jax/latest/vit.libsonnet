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
  local v3_32 = {
    accelerator: tpus.v3_32,
  },

  local vitTestScriptTemplate = |||
    set -x
    set -u
    set -e

    # .bash_logout sometimes causes a spurious bad exit code, remove it.
    rm .bash_logout

    pip install --upgrade pip
    %(installLatestJax)s
    %(maybeBuildJaxlib)s
    %(printDiagnostics)s

    pip install --upgrade clu %(extraDeps)s

    num_devices=`python3 -c "import jax; print(jax.device_count())"`
    if [ "$num_devices" = "1" ]; then
      echo "No TPU devices detected"
      exit 1
    fi

    git clone https://github.com/google-research/vision_transformer.git
    cd vision_transformer
    pip install -r vit_jax/requirements.txt

    export GCS_BUCKET=$(MODEL_DIR)
    export TFDS_DATA_DIR=$(TFDS_DIR)
    python3 -m vit_jax.main --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
      --workdir=$(MODEL_DIR) \
      --config.pretrained_dir="gs://vit_models/imagenet21k" \
      %(extraFlags)s
  |||,

  local vit = common.JaxTest + common.jaxlibLatest + common.alphaImage {
    local config = self,
    frameworkPrefix: 'flax-latest',
    modelName:: 'vit',
    extraDeps:: '',
    extraFlags:: '',
    testScript:: vitTestScriptTemplate % (self.scriptConfig {
                                            extraFlags: config.extraFlags,
                                            extraDeps: config.extraDeps,
                                          }),
  },

  local vit_pod = common.JaxPodTest + common.jaxlibLatest + common.alphaImage {
    local config = self,
    frameworkPrefix: 'flax-latest',
    modelName:: 'vit',
    extraDeps:: '',
    extraFlags:: '',
    testScript:: vitTestScriptTemplate % (self.scriptConfig {
                                            extraFlags: config.extraFlags,
                                            extraDeps: config.extraDeps,
                                          }),
  },

  configs: [
    vit + functional + v2_8,
    vit + convergence + v3_8,
  ],
}
