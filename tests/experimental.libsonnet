local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  TensorFlowTpuVmTest: {
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
          train+: {
            image: 'google/cloud-sdk',
            command: [
              'bash',
              '-c',
              |||
                set +e

                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) 'git clone https://github.com/tensorflow/models.git models/ && pip3 install -r models/official/requirements.txt && cd models && PYTHONPATH=. %s'
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % std.join(' ', config.command),
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
                    'ssh-keys': 'xl-ml-test:$(cat /scripts/id_rsa.pub)'
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

              sleep 30
            ||| % { acceleratorName: config.accelerator.name }),
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
}
