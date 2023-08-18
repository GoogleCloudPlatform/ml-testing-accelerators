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
  local llama2_google_next_inference_pretrained_models_eager = self.llama2_google_next_inference_pretrained_models_eager,
  llama2_google_next_inference_pretrained_models_eager:: common.PyTorchTest {
    local config = self,
    modelName: 'l2-e',
    paramsOverride:: {
      scriptPath: 'llama/example_text_completion.py',
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
  local llama2_google_next_inference_pretrained_models = self.llama2_google_next_inference_pretrained_models,
  llama2_google_next_inference_pretrained_models:: common.PyTorchTest {
    local config = self,
    modelName: 'l2',
    paramsOverride:: {
      scriptPath: 'llama/example_text_completion.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--ckpt_dir llama_2_model/llama-2-13b-dummy/',
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--max_seq_len 128 --max_batch_size 4',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_google_next_inference_fine_tuned_chat_models_eager = self.llama2_google_next_inference_fine_tuned_chat_models_eager,
  llama2_google_next_inference_fine_tuned_chat_models_eager:: common.PyTorchTest {
    local config = self,
    modelName: 'l2-c-e',
    paramsOverride:: {
      scriptPath: 'llama/example_chat_completion.py',
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
  local llama2_google_next_inference_fine_tuned_chat_models = self.llama2_google_next_inference_fine_tuned_chat_models,
  llama2_google_next_inference_fine_tuned_chat_models:: common.PyTorchTest {
    local config = self,
    modelName: 'l2-c',
    paramsOverride:: {
      scriptPath: 'llama/example_chat_completion.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--ckpt_dir llama_2_model/llama-2-13b-dummy/',
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--max_seq_len 512 --max_batch_size 4',
        '--mp True --dynamo True',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_stable_tokenizer = self.llama2_stable_tokenizer,
  llama2_stable_tokenizer:: common.PyTorchTest {
    local config = self,
    modelName: 'l2-t',
    paramsOverride:: {
      scriptPath: 'example.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--ckpt_dir llama_2_model/llama-2-13b-dummy/',
        '--tokenizer_path llama_2_model/tokenizer.model',
        '--mp True --dynamo True',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_stable_quant = self.llama2_stable_quant,
  llama2_stable_quant:: common.PyTorchTest {
    local config = self,
    modelName: 'l2-q',
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
        '--mp True --dynamo True',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_stable_quant_without_download = self.llama2_stable_quant_without_download,
  llama2_stable_quant_without_download:: common.PyTorchTest {
    local config = self,
    modelName: 'l2-q-w',
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
    modelName+: '-n-i',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        sudo apt update 
        sudo apt-get -y install libopenblas-dev
        pip install accelerate -U
        sudo apt update
        sudo apt-get -y install libopenblas-dev
        pip3 uninstall -y torch torch_xla
        pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
        pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
        pip3 uninstall -y libtpu-nightly
        pip3 install torch_xla[tpuvm] --user
        # show current path
        pwd
        ## /home/xl-ml-test
        ls
        ## pytorch snap
        git clone -b llama2-google-next-inference https://github.com/pytorch-tpu/llama.git
        cd llama
        # show current path
        pwd
        ## /home/xl-ml-test/llama
        ls
        ## CODE_OF_CONDUCT.md   CONTRIBUTING.md   LICENSE MODEL_CARD.md   README.md   Responsible-Use-Guide.pdf   USE_POLICY.md   download.sh   example_chat_completion.py
        ## example_text_completion.py   llama   requirements.txt   reshard_checkpoints.py   setup.py
        pip list | grep torch
        ## torch                    2.1.0
        ## torch-xla                2.1.0
        ## torchvision              0.16.0.dev20230817+cpu
        pip install -r requirements.txt
        pip install -e .
        pip list | grep torch
        ## torch                    2.1.0
        ## torch-xla                2.1.0
        ## torchvision              0.16.0.dev20230817+cpu
        # prepare data
        # show current path
        pwd
        ## /home/xl-ml-test/llama
        ls
        ## CODE_OF_CONDUCT.md   CONTRIBUTING.md   LICENSE   MODEL_CARD.md   README.md   Responsible-Use-Guide.pdf   USE_POLICY.md   download.sh   example_chat_completion.py
        ## example_text_completion.py   llama   llama.egg-info   requirements.txt   reshard_checkpoints.py   setup.py
        # --- wget https://storage.mtls.cloud.google.com/manfei_bucket/LLaMA2/llama_2_model.zip
        # show current path
        pwd
        ## /home/xl-ml-test/llama
        ls
        ## CODE_OF_CONDUCT.md   CONTRIBUTING.md   LICENSE   MODEL_CARD.md   README.md   Responsible-Use-Guide.pdf   USE_POLICY.md   download.sh   example_chat_completion.py
        ## example_text_completion.py   llama   llama.egg-info   llama_2_model.zip   requirements.txt   reshard_checkpoints.py   setup.py
        # --- sudo apt-get install unzip
        # --- unzip llama_2_model.zip
        wget -nv -O llama_2_model.zip https://storage.mtls.cloud.google.com/manfei_bucket/LLaMA2/llama_2_model.zip
        unzip -o llama_2_model.zip
        ## unzip:  cannot find zipfile directory in one of llama_2_model.zip or llama_2_model.zip.zip, and cannot find llama_2_model.zip.ZIP, period.
        # show current path
        pwd
        ## /home/xl-ml-test/llama
        ls
        ## CODE_OF_CONDUCT.md   CONTRIBUTING.md   LICENSE   MODEL_CARD.md   README.md   Responsible-Use-Guide.pdf   USE_POLICY.md   download.sh   example_chat_completion.py
        ## example_text_completion.py   llama   llama.egg-info   llama_2_model.zip   requirements.txt   reshard_checkpoints.py   setup.py
      |||,
    },
  },
  local stable = self.stable,
  stable:: common.PyTorchTpuVmMixin {
    modelName+: '-s',
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
  local xla = self.xla,
  xla:: common.PyTorchTpuVmMixin {
    modelName+: '-xla',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        sudo apt update 
        sudo apt-get -y install libopenblas-dev
        pip install accelerate -U
        sudo apt update
        sudo apt-get -y install libopenblas-dev
        pip3 uninstall -y torch torch_xla
        pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp310-cp310-linux_x86_64.whl
        pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl
        pip3 uninstall -y libtpu-nightly
        pip3 install torch_xla[tpuvm] --user
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    llama2_google_next_inference_pretrained_models + v4_8 + common.Functional + timeouts.Hours(3) + llama2_google_next_inference,
    // llama2_google_next_inference_fine_tuned_chat_models + v4_8 + common.Functional + timeouts.Hours(3) + llama2_google_next_inference + xla,
    llama2_stable_tokenizer + v4_8 + common.Functional + timeouts.Hours(3) + stable + xla,
    llama2_stable_quant + v4_8 + common.Functional + timeouts.Hours(3) + stable + xla,
    llama2_stable_quant_without_download + v4_8 + common.Functional + timeouts.Hours(3) + stable + xla,
  ],
}
