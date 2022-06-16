// Copyright 2022 Google LLC
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
  PyTorchTest:: common.PyTorchTest {
    frameworkPrefix: 'pt-r1.11',
    tpuSettings+: {
      softwareVersion: 'pytorch-1.11',
    },
    imageTag: 'r1.11',
  },
  PyTorchXlaDistPodTest:: common.PyTorchXlaDistPodTest {
    frameworkPrefix: 'pt-r1.11',
    tpuSettings+: {
      softwareVersion: 'pytorch-1.11',
    },
    imageTag: 'r1.11',
  },
  PyTorchGkePodTest:: common.PyTorchGkePodTest {
    frameworkPrefix: 'pt-r1.11',
    tpuSettings+: {
      softwareVersion: 'pytorch-1.11',
    },
    imageTag: 'r1.11',
  },
  Functional:: mixins.Functional {
    schedule: null,
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence {
    schedule: null,
  },
  PyTorchTpuVmMixin:: experimental.PyTorchTpuVmMixin {
    tpuSettings+: {
      softwareVersion: 'tpu-vm-pt-1.11',
      tpuVmPytorchSetup: |||
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-1.11-cp38-cp38-linux_x86_64.whl
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.11-cp38-cp38-linux_x86_64.whl
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-1.11-cp38-cp38-linux_x86_64.whl
        sudo pip3 install numpy
        sudo pip3 install mkl mkl-include
        sudo apt-get -y update
        sudo apt-get install -y libomp5
        git clone https://github.com/pytorch/pytorch.git -b release/1.11
        cd pytorch
        git clone https://github.com/pytorch/xla.git -b r1.11
      |||,
    },
  },
  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },

  // DEPRECATED: Use PyTorchTpuVmMixin instead
  tpu_vm_1_11_install: self.PyTorchTpuVmMixin.tpuSettings.tpuVmPytorchSetup,
}
