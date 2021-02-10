# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  TpuVmTest:: {
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

    publisherImage: null,
    volumeMap+: {
      scripts: volumes.MemoryVolumeSpec {
        name: 'scripts',
        mountPath: '/scripts',
      },
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local train = self,

            image: 'google/cloud-sdk',
            envMap+:: {
              'LOCAL_OUTPUT_DIR': '/tmp/model_dir',
            },

            local remoteScript = {
              dockerImage: config.image,
              dockerCommand: std.escapeStringBash(
                std.join(
                  ' ',
                  config.command,
                ),
              ),
            },
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u

                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo docker run -i --rm --privileged -v "/lib/libtpu.so:/lib/libtpu.so" -v "$(LOCAL_OUTPUT_DIR):$(LOCAL_OUTPUT_DIR)" --entrypoint "" %(dockerImage)s '%(dockerCommand)s
                exit_code=$?
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) 'gsutil -m cp -r $(LOCAL_OUTPUT_DIR) $(MODEL_DIR)'
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % remoteScript,
            ],
            lifecycle: cleanupHook,
            resources+: {
              // HACK: replace standard Cloud TPU resource.
              limits: {
                ['tpu.googleapis.com/v%s' % config.accelerator.version]: config.accelerator.size,
              },
            },
          },
        },
        initContainerMap+:: {
          'create-tpu': {
            image: 'google/cloud-sdk',
            command: utils.scriptCommand(|||
              project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              tpu_name=tpu-${POD_UID}
              ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""

              echo "
              curl -X DELETE \
                -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \
                https://tpu.googleapis.com/v2alpha1/projects/${project}/locations/${zone}/nodes/${tpu_name}
              " > /scripts/cleanup.sh

              curl -X POST \
                -H "Authorization: Bearer $(gcloud auth print-access-token)" \
                -H "Content-Type: application/json" \
                -d "{
                  accelerator_type: '%(acceleratorName)s',
                  runtime_version:'v2-alpha',
                  network_config: {enable_external_ips: true},
                  metadata: {
                    'ssh-keys': 'xl-ml-test:$(cat /scripts/id_rsa.pub)',
                    'startup-script': 'gcloud auth configure-docker && docker pull %(pullImage)s'
                  }
                }" https://tpu.googleapis.com/v2alpha1/projects/${project}/locations/${zone}/nodes?node_id=${tpu_name}

              echo "Waiting for TPU Pod ${tpu_name} to become ready..."
              while [[ ${health:-NONE} != "READY" ]];
                do sleep 10 && \
                health=$(gcloud \
                  --project=${project} \
                  compute \
                  tpus \
                  describe \
                  ${tpu_name} \
                  --zone=${zone} \
                  --format="value(state)") && \
                echo "Waiting for ready TPU (current state ${health:-NONE})...";
              done

              echo ${tpu_name} > /scripts/tpu_name
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(ipAddress)" > /scripts/tpu_ip

              sleep 180
            ||| % {acceleratorName: config.accelerator.name, pullImage: config.image}),
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
  TensorFlowTpuVmTest:: self.TpuVmTest {
    image: 'gcr.io/xl-ml-test/tensorflow-1vm:wcromar-20200210',
  },
}
