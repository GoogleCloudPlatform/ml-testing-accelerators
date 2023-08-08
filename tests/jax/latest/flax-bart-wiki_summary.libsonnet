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
  local hf_bart_common = self.hf_bart_common,
  hf_bart_common:: common.JaxTest + common.huggingFaceTransformer {
    local config = self,
    frameworkPrefix: 'flax.latest',
    modelName:: 'bart-wiki.summary',
    extraFlags:: [],
    setup: |||
      %(installPackages)s
      pip install -r examples/flax/summarization/requirements.txt
      %(verifySetup)s
    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
    runTest: |||
      export GCS_BUCKET=$(MODEL_DIR)
      python3 examples/flax/summarization/run_summarization_flax.py \
        --output_dir './bart-base-wiki' \
        --model_name_or_path facebook/bart-base \
        --tokenizer_name facebook/bart-base \
        --dataset_name 'wiki_summary' \
        --do_train \
        --do_eval \
        --do_predict \
        --predict_with_generate \
        --learning_rate 5e-5 \
        --warmup_steps 0 \
        --max_source_length 512 \
        --max_target_length 64 \
        %(extraFlags)s

      # Ignore CommandException for the rest workers in TPU pod
      gsutil -m cp -r ./bart-base-wiki $(MODEL_DIR) || exit 0
    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
  },

  local func = self.func,
  func:: mixins.Functional {
    extraFlags+:: ['--num_train_epochs 3'],
  },
  local conv = self.conv,
  conv:: mixins.Convergence {
    extraFlags+:: ['--num_train_epochs 30'],

    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            train_loss: {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 1.0,
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

  local v2_8 = self.v2_8,
  v2_8:: common.tpuVmBaseImage {
    accelerator: tpus.v2_8,
    extraFlags+:: ['--per_device_train_batch_size 32', '--per_device_eval_batch_size 32'],
  },
  local v2_32 = self.v2_32,
  v2_32:: common.tpuVmBaseImage {
    accelerator: tpus.v2_32,
    extraFlags+:: ['--per_device_train_batch_size 8', '--per_device_eval_batch_size 8'],
  },
  local v3_8 = self.v3_8,
  v3_8:: common.tpuVmBaseImage {
    accelerator: tpus.v3_8,
    extraFlags+:: ['--per_device_train_batch_size 32', '--per_device_eval_batch_size 32'],
  },
  local v3_32 = self.v3_32,
  v3_32:: common.tpuVmBaseImage {
    accelerator: tpus.v3_32,
    extraFlags+:: ['--per_device_train_batch_size 16', '--per_device_eval_batch_size 16'],
  },
  local v4_8 = self.v4_8,
  v4_8:: common.tpuVmV4Base {
    accelerator: tpus.v4_8,
    extraFlags+:: ['--per_device_train_batch_size 64', '--per_device_eval_batch_size 64'],
  },
  local v4_32 = self.v4_32,
  v4_32:: common.tpuVmV4Base {
    accelerator: tpus.v4_32,
    extraFlags+:: ['--per_device_train_batch_size 32', '--per_device_eval_batch_size 32'],
  },

  local func_tests = [
    hf_bart_common + func + v2_8,
    hf_bart_common + func + v3_8,
    hf_bart_common + func + v4_8,
  ],

  local conv_tests = [
    hf_bart_common + conv + v3_32,
    hf_bart_common + conv + v4_32,
  ],

  configs: func_tests + conv_tests,
}
