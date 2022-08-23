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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';
{
  local functional = mixins.Functional {
    extraFlags+: '--num_train_epochs 1',
    extraConfig:: 'default.py',
  },

  local convergence = mixins.Convergence {
    extraConfig:: 'default.py',
    extraFlags+: '--num_train_epochs 3 --learning_rate 2e-5 --eval_steps 500',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/accurary': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.85,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
  local mnli = {
    modelName+: '-mnli',
    extraFlags+: '--task_name mnli --max_seq_length 512 --eval_steps 1000 ',
  },
  local mrpc = {
    modelName+: '-mrpc',
    extraFlags+: '--task_name mrpc --max_seq_length 128 ',
  },

  local hfTestScriptTemplate = |||
    set -x
    set -u
    set -e

    # .bash_logout sometimes causes a spurious bad exit code, remove it.
    rm .bash_logout

    pip install --upgrade pip

    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .
    cd examples/flax
    pip install -r _tests_requirements.txt
    cd text-classification
    pip install -r requirements.txt
    pip install tensorflow
    pip install jax[tpu]>=0.2.16 \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html

    num_devices=`python3 -c "import jax; print(jax.device_count())"`
    if [ "$num_devices" = "1" ]; then
      echo "No TPU devices detected"
      exit 1
    fi

    export GCS_BUCKET=$(MODEL_DIR)
    export OUTPUT_DIR='./bert-glue'

    python3 run_flax_glue.py --model_name_or_path bert-base-cased \
      --output_dir ${OUTPUT_DIR} \
      --logging_dir ${OUTPUT_DIR} \
      --per_device_train_batch_size 4 \
      %(extraFlags)s
    gsutil -m cp -r ${OUTPUT_DIR} $(MODEL_DIR)
  |||,

  local bert = common.JaxTest + common.jaxlibLatest + common.tpuVmV4Base {
    local config = self,
    frameworkPrefix: 'flax-latest',
    modelName: 'hf-bert',
    extraDeps:: '',
    extraFlags:: '',
    testScript:: hfTestScriptTemplate % (self.scriptConfig {
                                           extraFlags: config.extraFlags,
                                           extraDeps: config.extraDeps,
                                         }),
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    bert + mnli + convergence + v4_8 + timeouts.Hours(10),
    bert + mrpc + convergence + v4_8 + timeouts.Hours(1),
    bert + mnli + functional + v4_8,
    bert + mrpc + functional + v4_8,
  ],
}
