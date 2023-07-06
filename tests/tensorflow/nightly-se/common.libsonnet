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
local metrics = import 'templates/metrics.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  HuggingFaceTransformer:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'nightly-se',
    tpuSettings+: {
      softwareVersion: 'nightly',
    },
    imageTag: 'nightly',
    script: {
      initialSetup:
        |||
          cd /tmp
          git clone https://github.com/huggingface/transformers.git
          cd transformers
          pip install .
          pip install -r examples/tensorflow/_tests_requirements.txt
        |||,
    },
  },
  ModelGardenTest:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'nightly-se',
    tpuSettings+: {
      softwareVersion: 'nightly',
    },
    imageTag: 'nightly',
    podTemplate+:: if config.accelerator.type == 'tpu' then
      {
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
                {
                  name: 'POD_UID',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: 'metadata.uid',
                    },
                  },
                },
              ],
              local tpuCreateSettings = {
                acceleratorName: std.escapeStringBash(config.accelerator.name),
                softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
                startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
                sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
                testName: std.strReplace(config.testName, '.', '-'),
              },
              command: [
                'python3',
                '-c',
                |||
                  import os
                  import tensorflow as tf
                  import urllib
                  import json
                  import cloud_tpu_client
                  import sys
                  print('python version: ' + str(sys.version))
                  print('tf_version: ' + str(tf.__version__))
                  #TODO(chandrasekhard):
                  # Add extra condition to fail if it picks stale image
                  print(str(tf.__file__))
                  ctc = cloud_tpu_client.Client(tpu=os.path.basename('$(TPU_NAME)'), zone=os.path.dirname('$(TPU_NAME)'))
                  ctc.wait_for_healthy()
                  ctc.configure_tpu_version('nightly', restart_type='always')
                  ctc.wait_for_healthy()
                  _VERSION_SWITCHER_ENDPOINT = 'http://{}:8475/requestversion'
                  url = _VERSION_SWITCHER_ENDPOINT.format(ctc.network_endpoints()[0]['ipAddress'])
                  req = urllib.request.Request(url)
                  resp = urllib.request.urlopen(req)
                  version_details = json.loads(resp.read())
                  print(version_details)
                |||,
              ],
            },
          },
        },
      }
    else
      {},
  },
  tpuVm:: experimental.TensorFlowTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: if config.accelerator.replicas == 1 then
        'v2-nightly'
      else
        'v2-nightly-pod',
    },
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'create-tpu'+: {
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
                gcloud alpha compute tpus tpu-vm delete -q ${tpu_name} --zone=${zone}
              " > /scripts/cleanup.sh

              curl -X POST \
                -H "Authorization: Bearer $(gcloud auth print-access-token)" \
                -H "Content-Type: application/json" \
                -d "{
                  accelerator_type: %(acceleratorName)s,
                  runtime_version: %(softwareVersion)s,
                  network_config: {enable_external_ips: true},
                  labels: {test_name: '%(testName)s' },
                  boot_disk: {source_image: 'projects/cloud-tpu-v2-images-dev/global/images/family/tpu-vm-tf-nightly'},
                  metadata: {
                    'ssh-keys': 'xl-ml-test:$(cat /scripts/id_rsa.pub)',
                    'startup-script': %(startupScript)s
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
              sleep %(sleepTime)d

              softwareVersion=%(softwareVersion)s
              if [[ ${softwareVersion: -3} == "pod" ]]; then
                 gcloud alpha compute tpus tpu-vm ssh ${tpu_name}  --zone=${zone} --project=${project}  --internal-ip --ssh-key-file=/scripts/id_rsa --worker=all --command "sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images-dev\/grpc_tpu_worker:se-nightly\"/' /etc/systemd/system/tpu-runtime.service"
                 gcloud alpha compute tpus tpu-vm ssh ${tpu_name}  --zone=${zone} --project=${project}  --internal-ip --ssh-key-file=/scripts/id_rsa --worker=all --command "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime"
              else
                 gcloud alpha compute tpus tpu-vm ssh ${tpu_name} --zone=${zone} --project=${project}  --command "sudo usermod -a -G docker $USER && gcloud auth configure-docker --quiet"
		 LIBTPU_DOCKER_IMAGE_PATH=gcr.io/cloud-tpu-v2-images-dev/libtpu:nightly
                 gcloud alpha compute tpus tpu-vm ssh ${tpu_name}  --zone=${zone} --project=${project}  --internal-ip --ssh-key-file=/scripts/id_rsa --worker=all --command "docker pull '${LIBTPU_DOCKER_IMAGE_PATH}' && docker create --name libtpu '${LIBTPU_DOCKER_IMAGE_PATH}' '/bin/bash' && sudo docker cp libtpu:libtpu.so /lib/libtpu.so"
              fi
            ||| % tpuCreateSettings),
          },
          'tpu-version': {
            image: 'google/cloud-sdk',
            command: null,
          },
        },
      },
    },
  },
  TfVisionTest:: self.ModelGardenTest + common.TfNlpVisionMixin {
    scriptConfig+: {
      runnerPath: 'official/vision/train.py',
    },
  },
  TfNlpTest:: self.ModelGardenTest + common.TfNlpVisionMixin {
    scriptConfig+: {
      runnerPath: 'official/nlp/train.py',
    },
  },
  TfRankingTest:: self.ModelGardenTest {
    paramsOverride:: {
      runtime: {
        distribution_strategy: error 'Must set `runtime.distribution_strategy`',
      },
      task: {
        train_data: {
          input_path: '$(CRITEO_DATA_DIR)/train/*',
          global_batch_size: 16384,
        },
        validation_data: {
          input_path: '$(CRITEO_DATA_DIR)/eval/*',
          global_batch_size: 16384,
        },
        model: {
          num_dense_features: 13,
          bottom_mlp: [512, 256, 64],
          embedding_dim: 64,
          top_mlp: [1024, 1024, 512, 256, 1],
          vocab_sizes: [
            39884406,
            39043,
            17289,
            7420,
            20263,
            3,
            7120,
            1543,
            63,
            38532951,
            2953546,
            403346,
            10,
            2208,
            11938,
            155,
            4,
            976,
            14,
            39979771,
            25641295,
            39664984,
            585935,
            12972,
            108,
            36,
          ],
        },
      },
      trainer: {
        use_orbit: true,
        validation_interval: 90000,
        checkpoint_interval: 270000,
        validation_steps: 5440,
        train_steps: 256054,
        optimizer_config: {
          embedding_optimizer: 'SGD',
          lr_config: {
            decay_exp: 1.6,
            decay_start_steps: 150000,
            decay_steps: 136054,
            learning_rate: 30,
            warmup_steps: 8000,
          },
        },
      },
    },
    command: [
      'python3',
      'official/recommendation/ranking/train.py',
      '--params_override=%s' % (std.manifestYamlDoc(self.paramsOverride) + '\n'),
      '--model_dir=$(MODEL_DIR)',
    ],
  },
  imagenet:: {
    scriptConfig+: {
      trainFilePattern: '$(IMAGENET_DIR)/train*',
      evalFilePattern: '$(IMAGENET_DIR)/valid*',
    },
  },
  coco:: {
    scriptConfig+: {
      trainFilePattern: '$(COCO_DIR)/train*',
      evalFilePattern: '$(COCO_DIR)/val*',
      paramsOverride+: {
        task+: {
          annotation_file: '$(COCO_DIR)/instances_val2017.json',
        },
      },
    },
  },
  local functional_schedule = '0 9 * * *',
  Functional:: mixins.Functional {
    schedule: if !(self.accelerator.type == 'tpu') || self.accelerator.name == 'v3-8' || self.accelerator.name == 'v4-8' then
      functional_schedule
    else
      null,
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
                  std_devs: 4.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
  // Override default schedule for Functional.
  RunNightly:: {
    schedule: functional_schedule,
  },
  Convergence:: mixins.Convergence {
    schedule: '0 5 * * 0,2,4',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
                  // TODO(wcromar): Tighten this restriction
                  std_devs: 2.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
}
