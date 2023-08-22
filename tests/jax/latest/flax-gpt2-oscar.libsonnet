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

local common = import '../common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local hf_gpt2_common = self.hf_gpt2_common,
  hf_gpt2_common:: common.JaxTest + common.huggingFaceTransformer {
    local config = self,
    frameworkPrefix: 'flax.latest',
    modelName:: 'gpt2-oscar',
    extraFlags:: [],
    setup: |||
      %(installPackages)s
      pip install -r examples/flax/language-modeling/requirements.txt
      %(verifySetup)s

      cd examples/flax/language-modeling
      gsutil cp -r gs://cloud-tpu-tpuvm-artifacts/config/xl-ml-test/jax/gpt2 .
    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
    runTest: |||
      python3 run_clm_flax.py \
        --output_dir=./gpt2 \
        --model_type=gpt2 \
        --config_name=./gpt2 \
        --tokenizer_name=./gpt2 \
        --dataset_name=oscar \
        --dataset_config_name=unshuffled_deduplicated_no \
        --do_train \
        --do_eval \
        --block_size=512 \
        --learning_rate=5e-3 \
        --warmup_steps=1000 \
        --adam_beta1=0.9 \
        --adam_beta2=0.98 \
        --weight_decay=0.01 \
        --overwrite_output_dir \
        --num_train_epochs=1 \
        --logging_steps=500 \
        --eval_steps=2500 \
         %(extraFlags)s
    ||| % (self.scriptConfig { extraFlags: std.join(' ', config.extraFlags) }),
  },

  local func = self.func,
  func:: mixins.Functional,

  local v4 = self.v4,
  v4:: common.tpuVmV4Base {
    extraFlags+:: ['--per_device_train_batch_size=64', '--per_device_eval_batch_size=64'],
  },

  local v4_8 = self.v4_8,
  v4_8:: v4 {
    accelerator: tpus.v4_8,
  },

  configs: [hf_gpt2_common + func + v4_8 + timeouts.Hours(2)],
}
