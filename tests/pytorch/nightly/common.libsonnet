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
local mixins = import 'templates/mixins.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

local version = 'nightly';
{
  PyTorchTest:: common.PyTorchTest {
    frameworkPrefix: 'pt-%(version)',
    tpuSettings+: {
      softwareVersion: 'pytorch-%(version)',
    },
    imageTag: '%(version)',
  },
  PyTorchXlaDistPodTest:: common.PyTorchXlaDistPodTest {
    frameworkPrefix: 'pt-%(version)',
    tpuSettings+: {
      softwareVersion: 'pytorch-%(version)',
    },
    imageTag: '%(version)',
  },
  PyTorchGkePodTest:: common.PyTorchGkePodTest {
    frameworkPrefix: 'pt-%(version)',
    tpuSettings+: {
      softwareVersion: 'pytorch-%(version)',
    },
    imageTag: '%(version)',
  },
  Functional:: mixins.Functional {
    schedule: '0 18 * * *',
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence {
    // Run at 22:00 PST on Monday and Thursday.
    schedule: '0 6 * * 1,3,6',
  },
  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },
  tpu_vm_nightly_install: |||
    sudo pip3 uninstall --yes torch torch_xla torchvision numpy
    sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly-cp38-cp38-linux_x86_64.whl numpy
    sudo pip3 install mkl mkl-include
    sudo apt-get -y update
    sudo apt-get install -y libomp5
    git clone https://github.com/pytorch/pytorch.git
    cd pytorch
    git clone https://github.com/pytorch/xla.git
  |||,
}
