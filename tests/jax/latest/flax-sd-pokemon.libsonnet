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
  local hf_sd_common = self.hf_sd_common,
  hf_sd_common:: common.JaxTest + common.huggingFaceDiffuser {
    local config = self,
    frameworkPrefix: 'flax.latest',
    modelName:: 'sd-pokemon',
    extraFlags:: [],
    testScript:: |||
      %(installPackages)s
      pip install -U -r examples/text_to_image/requirements_flax.txt
      %(verifySetup)s

      export GCS_BUCKET=$(MODEL_DIR)
      export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
      export dataset_name="lambdalabs/pokemon-blip-captions"
      python3 examples/text_to_image/train_text_to_image_flax.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --dataset_name=$dataset_name \
        --resolution=512 --center_crop --random_flip \
        --train_batch_size=8 \
        --num_train_epochs=10 \
        --learning_rate=1e-05 \
        --max_grad_norm=1 \
        --output_dir="./sd-pokemon-model" \
        --cache_dir /tmp \
        %(extraFlags)s

    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
  },

  local func = self.func,
  func:: mixins.Functional {
    extraFlags+:: ['--num_train_epochs 1'],
  },
  local conv = self.conv,
  conv:: mixins.Convergence {
    extraFlags+:: [],

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

  local v4 = self.v4,
  v4:: common.tpuVmV4Base {
    extraFlags+:: [],
  },

  local v4_8 = self.v4_8,
  v4_8:: v4 {
    accelerator: tpus.v4_8,
  },

  local func_tests = [
    hf_sd_common + func + v4_8,
  ],

  local conv_tests = [
    hf_sd_common + conv + v4 + v4_8,
  ],

  configs: func_tests + conv_tests,
}
