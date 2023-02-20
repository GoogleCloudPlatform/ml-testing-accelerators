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

local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  BaseTpuVmTest:: {
    local config = self,
    local cleanupHook = {
      preStop: {
        exec: {
          command: [
            'bash',
            '/scripts/cleanup.sh',
          ],
        },
      },
    },

    volumeMap+: {
      scripts: volumes.MemoryVolumeSpec {
        name: 'scripts',
        mountPath: '/scripts',
      },
    },

    testName+: '-1vm',

    tpuSettings+: {
      local tpuSettings = self,

      softwareVersion: if config.accelerator.version == 4 then
        'v2-nightly-tpuv4'
      else
        'v2-nightly',

      // Startup script in TPU VM metadata.
      tpuVmStartupScript: 'echo Running startup script',

      // Amount of time to sleep after TPU is READY.
      tpuVmCreateSleepSeconds:
        if config.accelerator.version <= 3 then
          60
        else
          90,

      // Additional arguments for test Docker container.
      tpuVmDockerArgs: '',
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            lifecycle: cleanupHook,
            resources+: {
              // HACK: remove standard Cloud TPU resource.
              local originalLimits = super.limits,
              limits: {
                [field]: originalLimits[field]
                for field in std.objectFields(originalLimits)
                if !std.startsWith(field, 'cloud-tpus.google.com')
              },
            },
          },
        },
        initContainerMap+:: {
          'create-tpu': {
            image: 'google/cloud-sdk',
            local tpuCreateSettings = {
              acceleratorName: std.escapeStringBash(config.accelerator.name),
              softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
              startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
              sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
              testName: std.strReplace(config.testName, '.', '-'),
            },
            command: utils.scriptCommand(|||
              project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              tpu_name=tpu-${POD_UID}
              ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""

              echo "
              curl -X DELETE \
                -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \
                https://tpu.googleapis.com/v2alpha1/projects/${project}/locations/${zone}/nodes/${tpu_name}
              sleep 60
              " > /scripts/cleanup.sh

              curl -X POST \
                -H "Authorization: Bearer $(gcloud auth print-access-token)" \
                -H "Content-Type: application/json" \
                -d "{
                  accelerator_type: %(acceleratorName)s,
                  runtime_version: %(softwareVersion)s,
                  network_config: {enable_external_ips: true},
                  boot_disk: {source_image: 'projects/cloud-tpu-v2-images-dev/global/images/tpu-vm-tf-2-12-0-20230218'},
                  metadata: {
                    'ssh-keys': 'xl-ml-test:$(cat /scripts/id_rsa.pub)',
                    'startup-script': %(startupScript)s,
                    'tensorflow-docker-url': 'gcr.io/cloud-tpu-v2-images-dev/grpc_tpu_worker:tf-2.12.0'
                  }
                }" https://tpu.googleapis.com/v2alpha1/projects/${project}/locations/${zone}/nodes?node_id=${tpu_name}

              echo "Waiting for TPU Pod ${tpu_name} to become ready..."
              timeout 10m bash -c -- "
              while [[ \${health:-NONE} != READY ]];
                do sleep 60 && \
                health=\$(gcloud \
                  --project=${project} \
                  compute \
                  tpus \
                  describe \
                  ${tpu_name} \
                  --zone=${zone} \
                  --format='value(state)') && \
                echo 'Waiting for ready TPU (current state \${health:-NONE})...';
              done
              "

              echo ${zone} > /scripts/zone
              echo ${tpu_name} > /scripts/tpu_name
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" > /scripts/tpu_ip
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --flatten="networkEndpoints[]" --format="csv[no-heading](networkEndpoints.ipAddress)" > /scripts/all_tpu_ips

              softwareVersion=%(softwareVersion)s
              if [[ ${softwareVersion: -3} == "pod" ]]; then
                 yes '' | gcloud compute config-ssh
                 sleep %(sleepTime)d
                 gcloud alpha compute tpus tpu-vm ssh ${tpu_name}  --zone=${zone} --project=${project}  --internal-ip --worker=all --command "sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images-dev\/grpc_tpu_worker:tf-2.12.0\"/' /etc/systemd/system/tpu-runtime.service"
                 gcloud alpha compute tpus tpu-vm ssh ${tpu_name}  --zone=${zone} --project=${project}  --internal-ip --worker=all --command "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime"
              fi
              gcloud alpha compute tpus tpu-vm ssh ${tpu_name}  --zone=${zone} --project=${project}  --internal-ip --worker=all --command "python3 -c 'import tensorflow as tf; print(tf.__version__)'"
              sleep %(sleepTime)d
            ||| % tpuCreateSettings),
            env: [
              {
                name: 'POD_UID',
                valueFrom: {
                  fieldRef: {
                    fieldPath: 'metadata.uid',
                  },
                },
              },
            ],
            volumeMounts: [
              {
                mountPath: '/scripts',
                name: 'scripts',
              },
            ],
          },
        },
      },
    },
  },
  // `BaseTpuVmMixin` is used to convert a 2VM target to 1VM.
  BaseTpuVmMixin:: self.BaseTpuVmTest {
    local config = self,

    // Disable retries
    jobTemplate+:: {
      spec+: {
        backoffLimit: 0,
      },
    },

    // TPU VM tests don't run the models directly
    cpu: 1,
    memory: '2Gi',

    // Pass TPU VM name to test container
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          train+: {
            image: 'google/cloud-sdk',
            envMap+:: {
              LOCAL_OUTPUT_DIR: '/tmp/model_dir',
              KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS: if config.accelerator.replicas == 1 then
                'local'
              else
                'tpu-$(POD_UID)',
            },
          },
        },
      },
    },
  },
}
