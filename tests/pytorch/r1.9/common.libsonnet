// Copyright 2021 Google LLC
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
local mixins = import 'templates/mixins.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  PyTorchTest:: common.PyTorchTest {
    frameworkPrefix: 'pt-r1.9',
    tpuSettings+: {
      softwareVersion: 'pytorch-1.9',
    },
    imageTag: 'r1.9',
  },
  PyTorchXlaDistPodTest:: common.PyTorchXlaDistPodTest {
    frameworkPrefix: 'pt-r1.9',
    tpuSettings+: {
      softwareVersion: 'pytorch-1.9',
    },
    imageTag: 'r1.9',
  },
  PyTorchGkePodTest:: common.PyTorchGkePodTest {
    frameworkPrefix: 'pt-r1.9',
    tpuSettings+: {
      softwareVersion: 'pytorch-1.9',
    },
    imageTag: 'r1.9',
  },
  Functional:: mixins.Functional {
    schedule: '30 5 * * *',
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence {
    // Run 3 times/week.
    schedule: '0 7 * * 1,3,5',
  },
  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },
}
