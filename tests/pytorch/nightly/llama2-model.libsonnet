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
    modelName: 'l2',
    paramsOverride:: {
      max_seq_len: 2048,
      max_gen_len: 1000,
      max_batch_size: 2,
      scriptPath: 'llama/example_text_completion.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        'True',
        '"/home/xl-ml-test/llama/7B"',
        '/home/xl-ml-test/spiece.model',
        '--max_seq_len=%d ' % config.paramsOverride.max_seq_len,
        '--max_gen_len=%d ' % config.paramsOverride.max_gen_len,
        '--max_batch_size=%d ' % config.paramsOverride.max_batch_size,
        '--dynamo=True',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    modelName+: '-n-i',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip3 uninstall torch torch_xla torchvision libtpu-nightly -y
        sudo apt-get update -y
        sudo apt-get install libomp5 -y
        pip3 install mkl mkl-include
        pip3 install tf-nightly tb-nightly tbp-nightly
        pip3 install numpy
        sudo apt-get install numactl -y
        sudo apt-get install libopenblas-dev -y
        pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly+20230821-cp310-cp310-linux_x86_64.whl
        pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly+20230821-cp310-cp310-linux_x86_64.whl
        pip3 install torch_xla[tpuvm]

        # install tokenizer model
        wget https://storage.googleapis.com/tpu-pytorch/lsiyuan-experiment/llama/spiece.model

        # git clone and build llama
        git clone --branch llama2-google-next-inference https://github.com/pytorch-tpu/llama.git
        cd llama
        pip3 install -r requirements.txt
        pip3 install -e .

        # 7B config
        mkdir 7B
        cd 7B/
        echo -e '{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}' >> params.json
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    llama2_google_next_inference_pretrained_models + v4_8 + common.Functional + timeouts.Hours(3) + pjrt,
  ],
}
