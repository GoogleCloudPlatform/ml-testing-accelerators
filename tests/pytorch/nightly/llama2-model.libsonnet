// Copyright 2020 Google LLC
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

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local llama2_google_next_inference_pretrained_models = self.llama2_google_next_inference_pretrained_models,
  llama2_google_next_inference_pretrained_models:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-model',
    paramsOverride:: {
      scriptPath: 'example_text_completion.py',
      trainCommand: [
        'torchrun --nproc_per_node 1',
        self.scriptPath,
        '--ckpt_dir llama_2_model/llama-2-13b-dummy/',
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--max_seq_len 128 --max_batch_size 4',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_google_next_inference_fine_tuned_chat_models = self.llama2_google_next_inference_fine_tuned_chat_models,
  llama2_google_next_inference_fine_tuned_chat_models:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-model-chat',
    paramsOverride:: {
      scriptPath: 'example_chat_completion.py',
      trainCommand: [
        'torchrun --nproc_per_node 1',
        self.scriptPath,
        '--ckpt_dir llama_2_model/llama-2-13b-dummy/',
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--max_seq_len 512 --max_batch_size 4',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_stable_tokenizer = self.llama2_stable_tokenizer,
  llama2_stable_tokenizer:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-model-tokenizer',
    paramsOverride:: {
      scriptPath: 'example.py',
      trainCommand: [
        'torchrun --nproc_per_node MP',
        self.scriptPath,
        '--ckpt_dir llama_2_model/llama-2-13b-dummy/',
        '--tokenizer_path llama_2_model/tokenizer.model',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_stable_quant = self.llama2_stable_quant,
  llama2_stable_quant:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-model-quant',
    paramsOverride:: {
      scriptPath: 'example_xla.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--ckpt_dir llama_2_model/tokenizer.model',
        '--max_seq_len 256',
        '--max_batch_size 1',
        '--temperature 0.8',
        '--mp True',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_stable_quant_without_download = self.llama2_stable_quant_without_download,
  llama2_stable_quant_without_download:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-model-quant-without',
    paramsOverride:: {
      scriptPath: 'example_xla.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--max_seq_len 256',
        '--max_batch_size 1',
        '--temperature 0.8',
        '--dim 4096',
        '--n_heads 32',
        '--n_layers 32',
        '--mp True',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_google_next_inference = self.llama2_google_next_inference,
  llama2_google_next_inference:: common.PyTorchTpuVmMixin {
    modelName+: '-next-infer',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        git clone -b llama2-google-next-inference https://github.com/pytorch-tpu/llama.git
        cd llama
        pip install -r requirements.txt
        pip install -e .
        # prepare data
        wget https://storage.mtls.cloud.google.com/manfei_bucket/LLaMA2/llama_2_model.zip
        sudo apt-get install unzip
        unzip llama_2_model.zip
      |||,
    },
  },
  local stable = self.stable,
  stable:: common.PyTorchTpuVmMixin {
    modelName+: '-stable',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        git clone -b stable https://github.com/pytorch-tpu/llama.git
        cd llama
        pip install -r requirements.txt
        pip install -e .
        # prepare data
        wget https://storage.mtls.cloud.google.com/manfei_bucket/LLaMA2/llama_2_model.zip
        sudo apt-get install unzip
        unzip llama_2_model.zip
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    llama2_google_next_inference_pretrained_models + v4_8 + common.Functional + timeouts.Hours(3) + llama2_google_next_inference,
    // llama2_google_next_inference_fine_tuned_chat_models + v4_8 + common.Functional + timeouts.Hours(3) + llama2_google_next_inference,
    llama2_stable_tokenizer + v4_8 + common.Functional + timeouts.Hours(3) + stable,
    llama2_stable_quant + v4_8 + common.Functional + timeouts.Hours(3) + stable,
    llama2_stable_quant_without_download + v4_8 + common.Functional + timeouts.Hours(3) + stable,
  ],
}
