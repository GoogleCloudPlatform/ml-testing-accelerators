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

local common = import '../common.libsonnet';
local experimental = import '../experimental.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  local Nightly = {
    frameworkPrefix: 'pt-nightly',
    tpuSettings+: {
      softwareVersion: 'pytorch-nightly',
    },
    imageTag: 'nightly_3.7',
  },
  PyTorchTest:: common.PyTorchTest + Nightly,
  PyTorchXlaDistPodTest:: common.PyTorchXlaDistPodTest + Nightly,
  PyTorchGkePodTest:: common.PyTorchGkePodTest + Nightly,
  Functional:: mixins.Functional {
    schedule: '0 7 * * *',
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence,
  PyTorchTpuVmMixin:: experimental.PyTorchTpuVmMixin {
    local config = self,

    tpuSettings+: {
      softwareVersion: if config.accelerator.version < 4 then
          'tpu-vm-base'
        else
          'tpu-vm-v4-base',
      tpuVmPytorchSetup: |||
        sudo pip3 uninstall --yes torch torch_xla torchvision libtpu-nightly numpy
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl numpy
        # Install corresponding libtpu-nightly
        sudo pip3 install torch_xla[tpuvm]
        sudo pip3 install mkl mkl-include cloud-tpu-client
        sudo apt-get -y update
        sudo apt-get install -y libomp5
        # No need to check out the PyTorch repository, but check out PT/XLA at
        # pytorch/xla anyway
        mkdir pytorch
        cd pytorch
        git clone https://github.com/pytorch/xla.git
      |||,
    },
  },
  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },

  // DEPRECATED: Use PyTorchTpuVmMixin instead
  tpu_vm_nightly_install: self.PyTorchTpuVmMixin.tpuSettings.tpuVmPytorchSetup,
}
