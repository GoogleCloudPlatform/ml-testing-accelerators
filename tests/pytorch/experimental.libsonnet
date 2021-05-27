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
                  'sudo mkdir /datasets && sudo mount 10.182.107.26:/pytorch_datasets /datasets'
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
}
