// Copyright 2021 Google LLC
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

local version = 'r1.10';
{
  PyTorchTest:: common.PyTorchTest {
    frameworkPrefix: 'pt-%s' % version,
    tpuSettings+: {
      softwareVersion: 'pytorch-%s' % version,
    },
    imageTag: version,
  },
  PyTorchXlaDistPodTest:: common.PyTorchXlaDistPodTest {
    frameworkPrefix: 'pt-%s' % version,
    tpuSettings+: {
      softwareVersion: 'pytorch-%s' % version,
    },
    imageTag: version,
  },
  PyTorchGkePodTest:: common.PyTorchGkePodTest {
    frameworkPrefix: 'pt-%s' % version,
    tpuSettings+: {
      softwareVersion: 'pytorch-%s' % version,
    },
    imageTag: version,
  },
  Functional:: mixins.Functional {
    schedule: '0 7 * * *',
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence {
    // Run 2 times/week.
    schedule: '0 7 * * 1,5',
  },
  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },
  tpu_vm_1_10_install: |||
    sudo bash /var/scripts/docker-login.sh
    sudo docker rm libtpu || true
    sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:pytorch-1.9 "/bin/bash"
    sudo docker cp libtpu:libtpu.so /lib
    sudo pip3 uninstall --yes torch torch_xla torchvision
    sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-1.10-cp38-cp38-linux_x86_64.whl
    sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-1.10-cp38-cp38-linux_x86_64.whl
    sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl
    git clone https://github.com/pytorch/pytorch.git -b release/1.10
    cd pytorch
    git clone https://github.com/pytorch/xla.git -b r1.10
    export XRT_TPU_CONFIG='localservice;0;localhost:51011'
    export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4'
  |||,
}
