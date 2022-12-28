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

local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local sd_model = common.PyTorchTest {
    local config = self,
    modelName: 'sd-model',
    paramsOverride:: {
      scriptPath: 'main_ll_profile.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--train',
        '--no-test',
        '--base=configs/latent-diffusion/cin-ldm-vq-f8-ss-ep2.yaml',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExports+: |||
        export XRT_TPU_CONFIG="localservice;0;localhost:51011"
        export TPU_NUM_DEVICES=4
        cd stable-diffusion/
      |||,
      tpuVmExtraSetup: |||
        git clone https://github.com/ssusie/stable-diffusion.git
        cd stable-diffusion
        pip install transformers==4.19.2 diffusers invisible-watermark
        pip install -e .
        pip install pytorch-lightning==1.4.2 torchmetrics==0.6.0
        pip install lmdb einops omegaconf
        pip install taming-transformers clip kornia==0.6 albumentations==0.4.3
        sudo apt-get update -y && sudo apt-get install libgl1 -y
        wget https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
        mv quantize.py ~/.local/lib/python3.8/site-packages/taming/modules/vqvae/

        # Setup data
        wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
        tar -xf  imagenette2.tgz
        mkdir -p ~/.cache/autoencoders/data/ILSVRC2012_train/data
        mkdir -p ~/.cache/autoencoders/data/ILSVRC2012_validation/data
        mv imagenette2/train/*  ~/.cache/autoencoders/data/ILSVRC2012_train/data
        mv imagenette2/val/* ~/.cache/autoencoders/data/ILSVRC2012_validation/data

        # Get first stage model
        wget -O models/first_stage_models/vq-f8/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
        cd  models/first_stage_models/vq-f8/
        unzip -o model.zip
        cd ~/stable-diffusion/
        mv device_parser.py ~/.local/lib/python3.8/site-packages/pytorch_lightning/utilities/device_parser.py
        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile
      |||,
    },
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    sd_model + v4_8 + common.Functional + timeouts.Hours(25) + tpuVm + mixins.Experimental,
  ],
}
