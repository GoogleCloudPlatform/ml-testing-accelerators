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
local utils = import 'templates/utils.libsonnet';

{
  PyTorchTpuVmMixin:: experimental.BaseTpuVmMixin {
    local config = self,

    // Don't need to mount datasets within Kubernetes for TPU VM.
    volumeMap+: { datasets: null },

    tpuSettings+: {
      tpuVmPytorchSetup: |||
        echo No PyTorch setup required.
      |||,
      tpuVmExtraSetup: |||
        echo No extra setup required.
      |||,
      // XRT_TPU_CONFIG set up by xla_dist on pods
      tpuVmExports:
        if config.accelerator.replicas == 1 then
          |||
            export XRT_TPU_CONFIG='localservice;0;localhost:51011'
          |||
        else
          '',
      tpuVmCreateSleepSeconds:
        if config.accelerator.replicas == 1 then
          super.tpuVmCreateSleepSeconds
        else
          180,
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              // Distribute command with xla_dist on pods
              testCommand: if config.accelerator.replicas == 1 then
                utils.toCommandString(config.command)
              else
                utils.toCommandString(
                  [
                    'python3',
                    '-m',
                    'torch_xla.distributed.xla_dist',
                    '--tpu=tpu-$(POD_UID)',
                    '--',
                  ] + config.command,
                ),
              pytorchSetup: config.tpuSettings.tpuVmPytorchSetup,
              extraSetup: config.tpuSettings.tpuVmExtraSetup,
              exports: config.tpuSettings.tpuVmExports,
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

                cat > workersetup.sh << TEST_SCRIPT_EOF
                sudo apt-get -y update
                sudo apt-get -y install nfs-common
                sudo mkdir /datasets && sudo mount $(PYTORCH_DATA_LOCATION) /datasets

                yes '' | gcloud compute config-ssh

                cd
                %(pytorchSetup)s

                cd
                %(extraSetup)s
                TEST_SCRIPT_EOF
                gcloud alpha compute tpus tpu-vm ssh xl-ml-test@$(cat /scripts/tpu_name) --zone=$(cat /scripts/zone) --ssh-key-file=/scripts/id_rsa --strict-host-key-checking=no --internal-ip --worker=all --command "$(cat workersetup.sh)"

                cat > testscript.sh << 'TEST_SCRIPT_EOF'
                %(exports)s
                %(testCommand)s
                TEST_SCRIPT_EOF
                gcloud alpha compute tpus tpu-vm ssh xl-ml-test@$(cat /scripts/tpu_name) --zone=$(cat /scripts/zone) --ssh-key-file=/scripts/id_rsa --strict-host-key-checking=no --internal-ip --worker=0 --command "$(cat testscript.sh)"

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
  PjRt:: {
    tpuSettings+: {
      tpuVmExports: |||
        export PJRT_DEVICE=TPU
      |||,
    },
  },

  // *TpuVmPodTest is deprecated. Use PyTorchTpuVmMixin instead
  PyTorchTpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
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
  PyTorch1_10TpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: 'v2-nightly',
      tpuVmStartupScript: |||
        sudo bash /var/scripts/docker-login.sh
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20211013-py3-none-any.whl
        sudo pip3 install torch==1.10.0
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl
        sudo pip3 install torchvision==0.11.1
        sudo pip3 install mkl mkl-include numpy
        sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.1 /usr/local/lib/libmkl_intel_lp64.so
        sudo ln -s /usr/local/lib/libmkl_intel_thread.so.1 /usr/local/lib/libmkl_intel_thread.so
        sudo ln -s /usr/local/lib/libmkl_core.so.1 /usr/local/lib/libmkl_core.so
        sudo apt-get -y update
        sudo apt-get install -y libomp5 nfs-common
        cd /usr/share
        git clone https://github.com/pytorch/pytorch.git -b release/1.10
        cd pytorch
        git clone https://github.com/pytorch/xla.git -b r1.10
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
  PyTorch1_11TpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: 'tpu-vm-pt-1.11',
      tpuVmStartupScript: |||
        sudo bash /var/scripts/docker-login.sh
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-1.11-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.11-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-1.11-cp38-cp38-linux_x86_64.whl
        sudo pip3 install mkl mkl-include numpy
        sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.1 /usr/local/lib/libmkl_intel_lp64.so
        sudo ln -s /usr/local/lib/libmkl_intel_thread.so.1 /usr/local/lib/libmkl_intel_thread.so
        sudo ln -s /usr/local/lib/libmkl_core.so.1 /usr/local/lib/libmkl_core.so
        sudo apt-get -y update
        sudo apt-get install -y libomp5 nfs-common
        gcloud compute config-ssh
        cd /usr/share
        git clone https://github.com/pytorch/pytorch.git -b r1.11
        cd pytorch
        git clone https://github.com/pytorch/xla.git -b release/1.11
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


  PyTorchNightlyTpuVmPodTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: 'v2-nightly',
      tpuVmStartupScript: |||
        sudo bash /var/scripts/docker-login.sh
        sudo pip3 uninstall --yes torch torch_xla torchvision numpy
        sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly-cp38-cp38-linux_x86_64.whl
        sudo pip3 install mkl mkl-include numpy
        sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.1 /usr/local/lib/libmkl_intel_lp64.so
        sudo ln -s /usr/local/lib/libmkl_intel_thread.so.1 /usr/local/lib/libmkl_intel_thread.so
        sudo ln -s /usr/local/lib/libmkl_core.so.1 /usr/local/lib/libmkl_core.so
        sudo apt-get -y update
        sudo apt-get install -y libomp5 nfs-common
        gcloud compute config-ssh
        cd /usr/share
        git clone https://github.com/pytorch/pytorch.git
        cd pytorch
        git clone https://github.com/pytorch/xla.git
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
