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
  BaseTpuVmSliceTest:: {
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

    testName+: '-%s-slices' % [config.tpuSettings.slices],

    tpuSettings+: {
      local tpuSettings = self,

      // Number of slices to be created
      slices: 2,

      softwareVersion: 'tpu-vm-v4-base',

      // Startup script in TPU VM metadata.
      tpuVmStartupScript: 'echo Running startup script',

      // Amount of time to sleep after TPU is READY.
      tpuVmCreateSleepSeconds: 60,

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
          'create-tpu-slices': {
            image: 'google/cloud-sdk',
            local tpuCreateSettings = {
              acceleratorName: std.escapeStringBash(config.accelerator.name),
              sliceCount: config.tpuSettings.slices,
              softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
              startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
              sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
              testName: std.strReplace(config.testName, '.', '-'),
            },
            command: utils.scriptCommand(|||
              set +x
              project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              tpu_name_prefix=tpu-${POD_UID}
              ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""

              echo "${project}:$(cat /scripts/id_rsa.pub)" > ssh-keys.txt
              echo %(startupScript)s > startup-script.txt

              echo %(sliceCount)s >> /scripts/slice_count
              for (( i=0; i < %(sliceCount)s; i++ )); do
                tpu_name=${tpu_name_prefix}-${i}
                echo "
                gcloud alpha compute tpus tpu-vm delete -q ${tpu_name} --zone=${zone}
                " > /scripts/cleanup_${i}.sh

                echo "
                bash /scripts/cleanup_${i}.sh
                " >> /scripts/cleanup.sh

                # Retry every 30 seconds for 10 minutes
                for j in {1..20}; do
                  set +e
                  gcloud alpha compute tpus tpu-vm create ${tpu_name} \
                    --accelerator-type=%(acceleratorName)s \
                    --version=%(softwareVersion)s  \
                    --metadata-from-file='ssh-keys=ssh-keys.txt,startup-script=startup-script.txt' \
                    --labels='test-name=%(testName)s' \
                    --zone=${zone}

                  exit_code=$?
                  set -e
                  test $exit_code = 0 && break || sleep 30;
                done

                if [ $exit_code -ne 0 ]; then
                  exit $exit_code
                fi

                echo ${tpu_name} >> /scripts/tpu_name_${i}

                if [ ${i} -eq 0 ]; then
                  gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" > /scripts/coordinator_ip
                fi
                gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" >> /scripts/tpu_ip_slice_${i}
                gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --flatten="networkEndpoints[]" --format="csv[no-heading](networkEndpoints.ipAddress)" >> /scripts/all_tpu_ips_slice_${i}
                wc -l < /scripts/all_tpu_ips_slice_${i} >> /scripts/worker_count_slice_${i}
              done

              sleep %(sleepTime)d

              COORDINATOR_IP=$(cat /scripts/coordinator_ip)
              SLICE_COUNT=$(cat /scripts/slice_count)

              for (( i=0; i < %(sliceCount)s; i++ )); do
                cat > set_mxla_flags.sh << SCRIPT_EOF
                echo "export MEGASCALE_COORDINATOR_ADDRESS=${COORDINATOR_IP}:8080" >> ~/.profile
                echo "export MEGASCALE_NUM_SLICES=${SLICE_COUNT}" >> ~/.profile
                echo "export MEGASCALE_SLICE_ID=${i}" >> ~/.profile
              SCRIPT_EOF

                gcloud alpha compute tpus tpu-vm ssh cloud-tpu-multipod-dev@$(cat /scripts/tpu_name_${i}) \
                --zone=${zone} \
                --ssh-key-file=/scripts/id_rsa \
                --strict-host-key-checking=no \
                --internal-ip \
                --worker=all \
                --command "$(cat set_mxla_flags.sh)"
              done

              echo ${zone} > /scripts/zone

              echo "LOGGER: TPU VMs created successfully."
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
  BaseTpuVmMixin:: self.BaseTpuVmSliceTest {
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
