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
  local hf_vit_common = self.hf_vit_common,
  hf_vit_common:: common.JaxTest + common.huggingFaceTransformer {
    local config = self,
    frameworkPrefix: 'flax.latest',
    modelName:: 'vit-imagenette',
    extraFlags:: [],
    setup: |||
      %(installPackages)s
      pip install -r examples/flax/vision/requirements.txt
      %(verifySetup)s

      wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
      tar -xvzf imagenette2.tgz
    ||| % self.scriptConfig,
    runTest: |||
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
    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
  },

  local func = self.func,
  func:: mixins.Functional {
    extraFlags+:: ['--model_type vit', '--num_train_epochs 5'],
  },
  local conv = self.conv,
  conv:: mixins.Convergence {
    extraFlags+:: ['--model_name_or_path google/vit-base-patch16-224-in21k', '--num_train_epochs 30'],

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

  local v2 = self.v2,
  v2:: common.tpuVmBaseImage {
    extraFlags+:: ['--per_device_train_batch_size 32', '--per_device_eval_batch_size 32'],
  },
  local v3 = self.v3,
  v3:: common.tpuVmBaseImage {
    extraFlags+:: ['--per_device_train_batch_size 32', '--per_device_eval_batch_size 32'],
  },
  local v4 = self.v4,
  v4:: common.tpuVmV4Base {
    extraFlags+:: ['--per_device_train_batch_size 64', '--per_device_eval_batch_size 64'],
  },

  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
  },
  local v2_32 = self.v2_32,
  v2_32:: {
    accelerator: tpus.v2_32,
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  local v3_32 = self.v3_32,
  v3_32:: {
    accelerator: tpus.v3_32,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },
  local v4_32 = self.v4_32,
  v4_32:: {
    accelerator: tpus.v4_32,
  },

  local func_tests = [
    hf_vit_common + func + v2 + v2_8,
    hf_vit_common + func + v3 + v3_8,
    hf_vit_common + func + v4 + v4_8,
  ],

  local conv_tests = [
    hf_vit_common + conv + v3 + v3_32,
    hf_vit_common + conv + v4 + v4_32,
  ],

  configs: func_tests + conv_tests,
}
