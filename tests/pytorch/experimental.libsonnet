// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';

{
  PyTorchTpuVmMixin:: experimental.BaseTpuVmMixin {
    local config = self,
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              testCommand:
                std.join(
                  ' ',
                  config.command,
                ),
            },
            args: null,
            // PyTorch tests are structured as bash scripts that run directly
            // on the Cloud TPU VM instead of using docker images.
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo apt-get -y update && sudo apt-get -y install nfs-common git google-perftools'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo mkdir /datasets && sudo mount $(PYTORCH_DATA_LOCATION) /datasets'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) << 'TEST_SCRIPT_EOF'
                  export XRT_TPU_CONFIG='localservice;0;localhost:51011'
                  export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4'
                  %(testCommand)s
                TEST_SCRIPT_EOF
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },

  PyTorchTpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: 'v2-nightly',
      tpuVmStartupScript: |||
        #! /bin/bash
        cd /usr/share
        git clone https://github.com/pytorch/xla.git -b r1.8.1
        sudo apt-get -y update
        sudo apt-get -y install nfs-common
        sudo mkdir /datasets
        sudo mount 10.182.107.26:/pytorch_datasets /datasets
        echo Done XLA startup script
      |||,
      tpuVmCreateSleepSeconds: 360,
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              testCommand:
                std.join(
                  ' ',
                  config.command,
                ),
            },
            args: null,
            // PyTorch tests are structured as bash scripts that run directly
            // on the Cloud TPU VM instead of using docker images.
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'ls'
                scp -i scripts/id_rsa /scripts/tpu_name xl-ml-test@$(cat /scripts/tpu_ip):~
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) << 'TEST_SCRIPT_EOF'
                  journalctl
                  cat ~/tpu_name
                  ls
                  echo | gcloud compute config-ssh
                  %(testCommand)s
                TEST_SCRIPT_EOF
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },
  PyTorch1_9TpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: 'v2-nightly',
      tpuVmStartupScript: |||
        sudo bash /var/scripts/docker-login.sh
        sudo docker rm libtpu || true
        sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:pytorch-1.9 "/bin/bash"
        sudo docker cp libtpu:libtpu.so /lib
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-1.9-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-1.9-cp38-cp38-linux_x86_64.whl numpy
        sudo pip3 install mkl mkl-include numpy
        sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.1 /usr/local/lib/libmkl_intel_lp64.so
        sudo ln -s /usr/local/lib/libmkl_intel_thread.so.1 /usr/local/lib/libmkl_intel_thread.so
        sudo ln -s /usr/local/lib/libmkl_core.so.1 /usr/local/lib/libmkl_core.so
        sudo apt-get -y update
        sudo apt-get install -y libomp5 nfs-common
        cd /usr/share
        git clone https://github.com/pytorch/pytorch.git -b release/1.9
        cd pytorch
        git clone https://github.com/pytorch/xla.git -b r1.9
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        sudo mkdir /datasets
        sudo mount 10.182.107.26:/pytorch_datasets /datasets
        echo Done XLA startup script
      |||,
      tpuVmCreateSleepSeconds: 360,
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              testCommand:
                std.join(
                  ' ',
                  config.command,
                ),
            },
            args: null,
            // PyTorch tests are structured as bash scripts that run directly
            // on the Cloud TPU VM instead of using docker images.
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'ls'
                scp -i scripts/id_rsa /scripts/tpu_name xl-ml-test@$(cat /scripts/tpu_ip):~
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) << 'TEST_SCRIPT_EOF'
                  journalctl
                  cat ~/tpu_name
                  ls
                  echo | gcloud compute config-ssh
                  %(testCommand)s
                TEST_SCRIPT_EOF
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },
  PyTorch1_9_1TpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: 'v2-nightly',
      tpuVmStartupScript: |||
        sudo bash /var/scripts/docker-login.sh
        sudo docker rm libtpu || true
        sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:pytorch-1.9.1 "/bin/bash"
        sudo docker cp libtpu:libtpu.so /lib
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-1.9.1-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9.1-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-1.9.1-cp38-cp38-linux_x86_64.whl numpy
        sudo pip3 install mkl mkl-include numpy
        sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.1 /usr/local/lib/libmkl_intel_lp64.so
        sudo ln -s /usr/local/lib/libmkl_intel_thread.so.1 /usr/local/lib/libmkl_intel_thread.so
        sudo ln -s /usr/local/lib/libmkl_core.so.1 /usr/local/lib/libmkl_core.so
        sudo apt-get -y update
        sudo apt-get install -y libomp5 nfs-common
        cd /usr/share
        git clone https://github.com/pytorch/pytorch.git -b release/1.9
        cd pytorch
        git clone https://github.com/pytorch/xla.git -b r1.9.1
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        sudo mkdir /datasets
        sudo mount 10.182.107.26:/pytorch_datasets /datasets
        echo Done XLA startup script
      |||,
      tpuVmCreateSleepSeconds: 360,
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              testCommand:
                std.join(
                  ' ',
                  config.command,
                ),
            },
            args: null,
            // PyTorch tests are structured as bash scripts that run directly
            // on the Cloud TPU VM instead of using docker images.
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'ls'
                scp -i scripts/id_rsa /scripts/tpu_name xl-ml-test@$(cat /scripts/tpu_ip):~
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) << 'TEST_SCRIPT_EOF'
                  journalctl
                  cat ~/tpu_name
                  ls
                  echo | gcloud compute config-ssh
                  %(testCommand)s
                TEST_SCRIPT_EOF
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },
}
