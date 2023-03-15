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
  local r2_0 = {
    frameworkPrefix: 'pt-2.0',
    tpuSettings+: {
      softwareVersion: 'pytorch-2.0',
    },
    imageTag: 'r2.0_3.8',
  },
  PyTorchTest:: common.PyTorchTest + r2_0 {
    local config = self,

    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': {
            image: config.podTemplate.spec.containerMap.train.image,
            env+: [
              {
                name: 'TPU_NAME',
                valueFrom: {
                  fieldRef: {
                    fieldPath: "metadata.annotations['name.cloud-tpus.google.com/train']",
                  },
                },
              },
            ],
            command: [
              'python3',
              '-c',
              |||
                import importlib_metadata
                import os
                import re

                import cloud_tpu_client

                requirements = importlib_metadata.requires('torch_xla')
                libtpu_pattern = r'libtpu-nightly ?@ https:\/\/storage.googleapis.com\/cloud-tpu-tpuvm-artifacts\/wheels\/libtpu-nightly\/libtpu_nightly-\d.\d.dev(\d{8})-\w+-\w+-\w+.whl'
                libtpu_matches = [
                  re.findall(libtpu_pattern, req)[0]
                  for req in requirements
                  if re.match(libtpu_pattern, req)
                ]
                assert len(libtpu_matches) == 1, f'{len(libtpu_matches)} matches in {requirements} (pattern: `{libtpu_pattern}`)'
                libtpu_date = libtpu_matches[0]
                print('libtpu date:', libtpu_date)

                ctc = cloud_tpu_client.Client(tpu=os.path.basename('$(TPU_NAME)'), zone=os.path.dirname('$(TPU_NAME)'))
                ctc.wait_for_healthy()
                ctc.configure_tpu_version(f'pytorch-2.0-dev{libtpu_date}', restart_type='always')
                ctc.wait_for_healthy()
              |||,
            ],
          },
        },
      },
    },
  },
  PyTorchXlaDistPodTest:: common.PyTorchXlaDistPodTest + r2_0,
  PyTorchGkePodTest:: common.PyTorchGkePodTest + r2_0,
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
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install torch==2.0 --index-url https://download.pytorch.org/whl/test/cpu
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-2.0-cp38-cp38-linux_x86_64.whl
        sudo pip3 install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20220930-py3-none-any.whl
        sudo pip3 install numpy
        sudo pip3 install mkl mkl-include cloud-tpu-client
        sudo apt-get -y update
        sudo apt-get install -y libomp5
        git clone https://github.com/pytorch/pytorch.git -b release/2.0
        cd pytorch
        git clone https://github.com/pytorch/xla.git -b r2.0
      |||,
    },
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': null,
        },
      },
    },
  },
  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },
  GpuMixin:: {
    local config = self,
    imageTag+: '_cuda_11.7',

    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': null,
        },
        containerMap+:: {
          train+: {
            envMap+: {
              GPU_NUM_DEVICES: '%d' % config.accelerator.count,
            },
          },
        },
      },
    },
  },

  tpu_vm_r2_0_install: self.PyTorchTpuVmMixin.tpuSettings.tpuVmPytorchSetup,
}
