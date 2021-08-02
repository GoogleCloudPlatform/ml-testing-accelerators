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
  LegacyTpuTest:: common.CloudAcceleratorTest {
    local config = self,

    image: 'gcr.io/xl-ml-test/tensorflow-tpu-1x',
  },
  ServingTest:: common.CloudAcceleratorTest {
    local config = self,
    loadTestImage: error 'must set loadTestImage.',
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
