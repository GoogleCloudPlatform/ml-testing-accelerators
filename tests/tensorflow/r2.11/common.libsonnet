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
local experimental = import 'experimental.libsonnet';
local metrics = import 'templates/metrics.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  ModelGardenTest:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'tf.r2.11.0',
    tpuSettings+: {
      softwareVersion: '2.11.1',
    },
    imageTag: 'r2.11.1',
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
                  print(str(tf.__file__))
                  ctc = cloud_tpu_client.Client(tpu=os.path.basename('$(TPU_NAME)'), zone=os.path.dirname('$(TPU_NAME)'))
                  ctc.wait_for_healthy()
                  ctc.configure_tpu_version('2.11.1', restart_type='always')
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
        'tpu-vm-tf-2.11.0'
      else
        'tpu-vm-tf-2.11.0-pod',
    },
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
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
  local functional_schedule = '0 10 * * *',
  Functional:: mixins.Functional {
    schedule: functional_schedule,
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
    schedule: '0 13 * * 0,4',
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
