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
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local command_common = |||
    git clone https://github.com/ssusie/stable-diffusion.git
    cd stable-diffusion
    pip install transformers==4.19.2 diffusers invisible-watermark
    pip install -e .
    pip install pytorch-lightning==1.4.2 torchmetrics==0.6.0
    pip install lmdb einops omegaconf
    pip install taming-transformers clip kornia==0.6 albumentations==0.4.3
    sudo apt-get install libgl1 -y
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

    python3 main_ll_profile.py \
    --train --no-test \
  |||,
  local base_metrics = common.Convergence {
    modelName: 'sd-base-metrics',
    command: utils.scriptCommand(
      |||
        %(common)s --base configs/latent-diffusion/cin-ldm-vq-f8-ss.yaml
      ||| % { common: command_common}
    ),
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/loss': {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 0.17,
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
  local training_time = common.Convergence {
    command: utils.scriptCommand(
      |||
        %(command_common)s --base configs/latent-diffusion/cin-ldm-vq-f8-ss.yaml \
          2>&1 | tee training_logs.txt
        train_time=$(
          cat training_logs.txt | grep 'Training time:' |  \
          grep -oP '\K[+-]?([0-9]*[.])?[0-9]+'
        )
        echo 'Training time is' $train_time
        test $(echo $train_time'<'100000 | bc -l) -eq 1  # assert model trainig time is less then 100000 seconds
      ||| % command_common
    ),
  },
  local functional = common.Functional {
    command: utils.scriptCommand(
      |||
        %(common)s --base configs/latent-diffusion/cin-ldm-vq-f8-ss-ep2.yaml
      ||| % {common: command_common}
    ),
  },
  local sd_model = common.PyTorchTest {
    modelName: 'sd-model',
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: '12.0',
                memory: '80Gi',
              },
            },
          },
        },
      },
    },
  },
  local tpuVm = common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExports+: |||
        export XRT_TPU_CONFIG="localservice;0;localhost:51011"
        export TPU_NUM_DEVICES=4
      |||,
      tpuVmExtraSetup: |||
        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile
      |||,
    },
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    sd_model + v4_8 + training_time + timeouts.Hours(30) + tpuVm,
    sd_model + v4_8 + base_metrics + timeouts.Hours(30) + tpuVm,
    sd_model + v4_8 + functional + timeouts.Hours(25) + tpuVm,
  ],
}
