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
  local command_common = |||
    git clone https://github.com/pytorch-tpu/training.git unet3d_test
    pip3 install -r unet3d_test/image_segmentation/pytorch/requirements.txt
    pip3 install tqdm

    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    cd -

    python3 unet3d_test/image_segmentation/pytorch/main.py --data_dir /kits19 \
    --epochs 501 \
    --evaluate_every 250 \
    --start_eval_at 250 \
    --quality_threshold 0.2 \
    --batch_size 1 \
    --optimizer sgd \
    --ga_steps 1 \
    --learning_rate 0.8 \
    --seed 0 \
    --lr_warmup_epochs 0 \
    --input_shape 128 128 128 \
    --debug \
    --device xla
  |||,
  local unet3d = self.unet3d,
  unet3d:: common.PyTorchTest {
    modelName: 'unet3d',

    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '9.0',
    memory: '30Gi',
  },
  local conv = self.conv,
  conv:: common.Convergence {
    command: utils.scriptCommand(
      |||
        %(command_common)s \
          2>&1 | tee training_logs.txt
        acc=$(
          cat training_logs.txt | grep 'eval_accuracy' | \
          tail -1 | grep -oP '"value": \K[+-]?([0-9]*[.])?[0-9]+'
        )
        echo 'Final UNet3D model accuracy is' $acc
        test $(echo $acc'>'0.2 | bc -l) -eq 1  # assert model accuracy is higher than 0.2
      ||| % command_common
    ),
  },
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  local tpuVm = self.tpuVm,
  tpuVm:: common.PyTorchTpuVmMixin {

    tpuSettings+: {

      tpuVmExtraSetup: |||

        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile

      |||,

    },

  },
  configs: [
    unet3d + v3_8 + conv + timeouts.Hours(25) + tpuVm + mixins.Experimental,
  ],
}
