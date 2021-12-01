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
local metrics = import 'templates/metrics.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  ModelGardenTest:: common.CloudAcceleratorTest {
    local config = self,

    image: 'gcr.io/xl-ml-test/tensorflow',
    cpu: 2,
    memory: '20G',
    volumeMap+: {
      dshm: volumes.MemoryVolumeSpec {
        name: 'dshm',
        mountPath: '/dev/shm',
      },
    },
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              TF_ENABLE_LEGACY_FILESYSTEM: '1',
            },
          },
        },
      },
    },
  },
  TfNlpVisionMixin:: {
    local config = self,
    additionalOverrides:: {},

    scriptConfig:: {
      runnerPath: error 'Must define `runnerPath.`',
      experiment: error 'Must define `experiment`',
      configFiles: [],
      trainFilePattern: '',
      evalFilePattern: '',
      paramsOverride:: {
        runtime: {
          distribution_strategy: 'tpu',
        },
        task: {
          train_data: {
            tfds_data_dir: '$(TFDS_DIR)',
            input_path: '%s' % config.scriptConfig.trainFilePattern,
          },
          validation_data: {
            tfds_data_dir: '$(TFDS_DIR)',
            input_path: '%s' % config.scriptConfig.evalFilePattern,
          },
        },
      },
    },
    command: [
      'python3',
      '%(runnerPath)s' % config.scriptConfig,
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--experiment=%(experiment)s' % config.scriptConfig,
      '--mode=train_and_eval' % config.scriptConfig,
      '--model_dir=$(MODEL_DIR)',
      '--params_override=%(paramsOverride)s' % config.scriptConfig {
        paramsOverride: std.manifestYamlDoc(config.scriptConfig.paramsOverride) + '\n',
      },
    ] + ['--config_file=%s' % configFile for configFile in config.scriptConfig.configFiles],
  },
  LegacyTpuTest:: common.CloudAcceleratorTest {
    local config = self,
    image: 'gcr.io/xl-ml-test/tensorflow-tpu-1x',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          merge_runs: true,
        },
      },
    },
  },
  ServingTest:: common.CloudAcceleratorTest {
    local config = self,
    servingConfig:: {
      gcsDir: error 'must set gcsDir.',
      dataType: error 'must set dataType.',
      batchSize: error 'must set batchSize.',
      servingImage: error 'must set servingImage.',
      model: error 'must set model.',
    },
    modelName: '%(model)s-serving' % config.servingConfig,
  },
}
