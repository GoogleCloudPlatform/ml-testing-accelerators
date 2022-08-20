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
  local hf_vit_common = common.JaxTest + common.jaxlibLatest {
    local config = self,
    frameworkPrefix: 'flax-latest',
    modelName:: 'hf-vit',
    extraFlags:: '',
    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      pip install --upgrade pip
      git clone https://github.com/huggingface/transformers.git
      cd transformers && pip install .
      pip install -r examples/flax/_tests_requirements.txt
      pip install -r examples/flax/vision/requirements.txt
      pip install --upgrade huggingface-hub urllib3 zipp

      %(testEnvWorkarounds)s
      %(installLatestJax)s

      python3 -c 'import flax; print("flax version:", flax.__version__)'
      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      if [ "$num_devices" = "1" ]; then
        echo "No TPU devices detected"
        exit 1
      fi

      wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
      tar -xvzf imagenette2.tgz

      export GCS_BUCKET=$(MODEL_DIR)
      python3 examples/flax/vision/run_image_classification.py \
        --output_dir './vit-imagenette' \
        --train_dir='imagenette2/train' \
        --validation_dir='imagenette2/val' \
        --learning_rate 1e-3 \
        --preprocessing_num_workers 32 \
        %(extraFlags)s

      # Ignore CommandException for the rest workers in TPU pod
      gsutil -m cp -r ./vit-imagenette $(MODEL_DIR) || exit 0
    ||| % (self.scriptConfig { extraFlags: config.extraFlags }),
  },

  local hf_vit_func_v2_v3 = common.tpuVmBaseImage + mixins.Functional {
    extraFlags:: '--model_type vit --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32',
  },

  local hf_vit_func_v4 = common.tpuVmV4Base + mixins.Functional {
    extraFlags:: '--model_type vit --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64',
  },

  local hf_vit_conv_v4 = common.tpuVmV4Base + mixins.Convergence {
    extraFlags:: '--model_name_or_path google/vit-base-patch16-224-in21k --num_train_epochs 30 --per_device_train_batch_size 64 --per_device_eval_batch_size 64',

    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            eval_accuracy: {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 0.98,
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

  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  local v4_32 = {
    accelerator: tpus.v4_32,
  },

  local func_tests_v2_v3 = [
    hf_vit_common + hf_vit_func_v2_v3 + accelerator
    for accelerator in [v2_8, v2_32, v3_8, v3_32]
  ],
  local func_tests_v4 = [
    hf_vit_common + hf_vit_func_v4 + accelerator
    for accelerator in [v4_8, v4_32]
  ],
  local conv_tests_v4 = [hf_vit_common + hf_vit_conv_v4 + v4_32],

  configs: func_tests_v2_v3 + func_tests_v4 + conv_tests_v4,
}
