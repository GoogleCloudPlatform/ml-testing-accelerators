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
  runFlaxLatest:: common.JaxTest + common.jaxlibLatest + common.tpuVmBaseImage {
    local config = self,

    frameworkPrefix: 'flax.latest',
    extraDeps:: [],
    extraFlags:: [],

    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      pip install --upgrade pip
      pip install --upgrade clu %(extraDeps)s

      %(installLatestJax)s
      %(maybeBuildJaxlib)s
      %(printDiagnostics)s

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      if [ "$num_devices" = "1" ]; then
        echo "No TPU devices detected"
        exit 1
      fi

      git clone https://github.com/google/flax
      cd flax
      pip install -e .
      cd examples/%(folderName)s

      export GCS_BUCKET=$(MODEL_DIR)
      export TFDS_DATA_DIR=$(TFDS_DIR)

      python3 main.py --workdir=$(MODEL_DIR)  --config=configs/%(extraConfig)s %(extraFlags)s
    ||| % (self.scriptConfig {
             folderName: config.folderName,
             modelName: config.modelName,
             extraDeps: std.join(' ', config.extraDeps),
             extraConfig: config.extraConfig,
             extraFlags: std.join(' ', config.extraFlags),
           }),
  },

  hfBertCommon:: common.JaxTest + common.huggingFace {
    local config = self,
    frameworkPrefix: 'flax.latest',
    modelName:: 'bert-glue',
    extraFlags:: [],
    testScript:: |||
      %(installPackages)s
      pip install -r examples/flax/text-classification/requirements.txt
      %(verifySetup)s

      export GCS_BUCKET=$(MODEL_DIR)
      export OUTPUT_DIR='./bert-glue'

      python3 examples/flax/text-classification/run_flax_glue.py --model_name_or_path bert-base-cased \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size 4 \
        %(extraFlags)s

      # Upload files from worker 0, and ignore CommandException for the rest workers in TPU pod
      gsutil -m cp -r ${OUTPUT_DIR} $(MODEL_DIR) || exit 0
    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
  },

  tpuEmbeddingCommon:: common.JaxTest {
        local config = self,
        frameworkPrefix: 'jax.tpu.embedding',
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
}
