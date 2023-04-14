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

local tpus = import 'templates/tpus.libsonnet';

{
  local clusters = self,

  defaultCluster: 'us-central1',
  clusterAccelerators: {
    'us-central1': [],
    'us-central2': [
      tpus.v4_8,
      tpus.v4_16,
      tpus.v4_32,
    ],
    'europe-west4': [
      tpus.v2_32,
      tpus.v3_32,
    ],
  },
  acceleratorClusters:: std.foldl(
    function(result, cluster) result + {
      [accelerator.name]: cluster
      for accelerator in clusters.clusterAccelerators[cluster]
    },
    std.objectFields(clusters.clusterAccelerators),
    {},
  ),
}
