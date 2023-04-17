// Copyright 2023 Google LLC
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
  local tpu_embedding = self.tpu_embedding,
  tpu_embedding:: common.JaxTest {
    local config = self,
    frameworkPrefix: 'jax-tpu-embedding',
    extraFlags:: [],
    testCommand:: error 'Must define `testCommand`',
    testScript:: |||
      pip install --upgrade 'jax[tpu]==0.4.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      pip install flax==0.6.7
      gsutil cp gs://cloud-tpu-tpuvm-artifacts/tensorflow/20230214/tf_nightly-2.13.0-cp38-cp38-linux_x86_64.whl .
      pip install tf_nightly-2.13.0-cp38-cp38-linux_x86_64.whl
      git clone https://github.com/jax-ml/jax-tpu-embedding.git
      %(testCommand)s
    ||| % config.testCommand,
  },
  local tpu_embedding_pjit = self.tpu_embedding_pjit,
  tpu_embedding_pjit:: tpu_embedding {
    local config = self,
    modelName: 'pjit',
    extraFlags:: [],
    testCommand: |||
      source ~/jax-tpu-embedding/jax_tpu_embedding/scripts/export_tpu_pod_env.sh && PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding python3 ~/jax-tpu-embedding/jax_tpu_embedding/examples/singlehost_pjit_example.py
    |||,
  },
  local tpu_embedding_pmap = self.tpu_embedding_pmap,
  tpu_embedding_pmap:: tpu_embedding {
    local config = self,
    modelName: 'pmap',
    extraFlags:: [],
    testCommand: |||
      source ~/jax-tpu-embedding/jax_tpu_embedding/scripts/export_tpu_pod_env.sh && PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding python3 ~/jax-tpu-embedding/jax_tpu_embedding/examples/pmap_example.py
    |||,
  },
  local func = self.func,
  func:: mixins.Functional {
    extraFlags+:: ['--num_train_epochs 5'],
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },

  configs: [
    tpu_embedding_pjit + func + v4_8,
    tpu_embedding_pmap + func + v3_8,
  ],
}
