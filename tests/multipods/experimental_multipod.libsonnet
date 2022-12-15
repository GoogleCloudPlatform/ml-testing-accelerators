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

      softwareVersion: 'tpu-vm-base-gvnic',

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
              acceleratorName: config.accelerator.name,
              sliceCount: config.tpuSettings.slices,
              softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
              startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
              sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
              testName: std.strReplace(config.testName, '.', '-'),
              tpuExists: config.tpuExists,
              tpuPrefix: config.tpuPrefix,
              userName: config.userName,
            },
            command: utils.scriptCommand(|||
              set +x
              project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              tpu_name_prefix=tpu-${POD_UID}
              if [ %(tpuExists)s = true ]; then
                tpu_name_prefix=%(tpuPrefix)s
              fi
              ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""

              echo "${project}:$(cat /scripts/id_rsa.pub)" > ssh-keys.txt
              echo %(startupScript)s > startup-script.txt

              echo %(sliceCount)s >> /scripts/slice_count
              if [ %(tpuExists)s = false ]; then
                for (( i=0; i < %(sliceCount)s; i++ )); do
                  tpu_name_delete=${tpu_name_prefix}-${i}
                  echo "
                  gcloud alpha compute tpus tpu-vm delete -q ${tpu_name_delete} --zone=${zone} --project=${project}
                  " > /scripts/cleanup_${i}.sh
                  echo "
                  bash /scripts/cleanup_${i}.sh
                  " >> /scripts/cleanup.sh
                done
              else
                echo "
                true
                " >> /scripts/cleanup.sh
              fi
              delete_tpus() {
                echo -e "\n\nDeleting TPUs..."
                for tpu_id in "${TPU_LIST[@]}"; do
                  echo -e "\n${tpu_id} is being deleted."
                  gcloud alpha compute tpus tpu-vm delete -q "${tpu_id}" --zone=${zone} --project=${project}
                  if [[ $? -ne 0 ]]; then
                    echo "Failed to delete the TPU ${TPU_NAME}. Delete it manually."
                    exit 1
                  fi
                done
              }
              create_tpu() {
                echo "Create TPU called"
                TPU_NAME=$1
                SLICE_ID=$2
                # Retry every 30 seconds for 10 minutes
                for j in {1..20}; do
                  set +e
                  gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
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
                  echo "TPU VM with name ${TPU_NAME} failed to create. So exiting the setup."
                  delete_tpus
                  exit $exit_code
                fi
                echo -e "Slice_${SLICE_ID}: TPU VM ${TPU_NAME} successfully created."
                TPU_CREATED=true
              }
              create_tpu_slice_environment() {
                echo -e "\n\nSetting %(sliceCount)s TPU Slices with %(acceleratorName)s in each slice..."
                for (( i=0; i < %(sliceCount)s; i++ )); do
                  TPU_NAME=${tpu_name_prefix}-${i}
                  tpu_exist_with_same_type=false
                  tpu_exist_with_diff_type=false
                  echo "$TPU_NAME, $zone, $project, $(gcloud compute tpus list --zone=${zone} --project=${project} | grep "^$TPU_NAME ")"
                  if [[ -z "$(gcloud compute tpus list --zone=${zone} --project=${project} | grep "^$TPU_NAME ")" ]]; then
                    list_of_tpu_with_same_name=''
                  else
                    list_of_tpu_with_same_name=$(gcloud compute tpus list --zone=${zone} --project=${project} | grep "^$TPU_NAME ")
                  fi
                  if [[ ! -z "$(gcloud compute tpus list --zone=${zone} --project=${project} | grep "^$TPU_NAME ")" ]]; then
                    list_of_tpu_with_same_type=$(echo "$list_of_tpu_with_same_name" | grep "%(acceleratorName)s")
                    echo "$list_of_tpu_with_same_type"
                    if [[ ! -z "$list_of_tpu_with_same_type" ]]; then
                      tpu_exist_with_same_type=true
                    else
                      tpu_exist_with_diff_type=true
                    fi
                  fi
                  echo "$TPU_NAME, $tpu_exist_with_same_type, $tpu_exist_with_diff_type"
                  if [[ %(tpuExists)s = true ]]; then
                    if [[ "$tpu_exist_with_same_type" = false ]]; then
                      if [[ "$tpu_exist_with_diff_type" = false ]]; then
                        echo -e "\nYou chooses to use existing TPU. But TPU with name $TPU_NAME doesn't exist!"
                      else
                        echo -e "\nTPU with name $TPU_NAME already exists but with different configuration. So exiting."
                      fi
                      exit 1
                    fi
                  else
                    if [[ "$tpu_exist_with_same_type" = true ]] || [[ "$tpu_exist_with_diff_type" = true ]]; then
                      echo -e "\nTPU with name $TPU_NAME already exists and you choose USE_EXISTING_TPUS=%(tpuExists)s. So exiting."
                      exit 1
                    fi
                    create_tpu "$TPU_NAME" $i
                  fi
                  TPU_LIST+=(${TPU_NAME})
                  echo ${TPU_NAME} >> /scripts/tpu_name_${i}
                  if [ ${i} -eq 0 ]; then
                    gcloud compute tpus describe ${TPU_NAME} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" > /scripts/coordinator_ip
                  fi
                  gcloud compute tpus describe ${TPU_NAME} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" >> /scripts/tpu_ip_slice_${i}
                  gcloud compute tpus describe ${TPU_NAME} --project=${project} --zone=${zone} --flatten="networkEndpoints[]" --format="csv[no-heading](networkEndpoints.ipAddress)" >> /scripts/all_tpu_ips_slice_${i}
                  wc -l < /scripts/all_tpu_ips_slice_${i} >> /scripts/worker_count_slice_${i}
                done
                if [[ "$TPU_CREATED" = false ]]; then
                  echo -e "\nUsing already created %(sliceCount)s TPU Slices with %(acceleratorName)s in each slice..."
                fi
              }
              TPU_CREATED=false
              create_tpu_slice_environment
              echo "$TPU_LIST"
              sleep %(sleepTime)d

              COORDINATOR_IP=$(cat /scripts/coordinator_ip)
              SLICE_COUNT=$(cat /scripts/slice_count)

              for (( i=0; i < %(sliceCount)s; i++ )); do
                cat > set_mxla_flags.sh << SCRIPT_EOF
                echo "export MEGASCALE_COORDINATOR_ADDRESS=${COORDINATOR_IP}:8080" >> ~/.profile
                echo "export MEGASCALE_NUM_SLICES=${SLICE_COUNT}" >> ~/.profile
                echo "export MEGASCALE_SLICE_ID=${i}" >> ~/.profile
                echo "export MEGASCALE_TRANSPORT_TYPE=\"grpc\"" >> ~/.profile
                echo "export MEGASCALE_PORT=8080" >> ~/.profile
                echo "export MEGASCALE_AUTHENTICATION=\"insecure\"" >> ~/.profile
              SCRIPT_EOF
                echo $(cat /scripts/tpu_name_${i})
                echo "$(cat set_mxla_flags.sh)"
                gcloud alpha compute tpus tpu-vm ssh %(userName)s@$(cat /scripts/tpu_name_${i}) \
                --zone=${zone} \
                --ssh-key-file=/scripts/id_rsa \
                --strict-host-key-checking=no \
                --internal-ip \
                --worker=all \
                --command "$(cat set_mxla_flags.sh)"
              done

              echo ${zone} > /scripts/zone

              echo "LOGGER: TPU VMs setup successful."
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

