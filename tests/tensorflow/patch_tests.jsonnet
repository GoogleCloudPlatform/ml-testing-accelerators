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

local clusters = import '../clusters.jsonnet';
local targets = import 'targets.jsonnet';
local utils = import 'templates/utils.libsonnet';

// Convert rX.Y.Z to rX.Y
local minorVersionFromPatchVersion(patchVersion) =
  local segments = std.splitLimit(patchVersion, '.', 3);
  segments[0] + '.' + segments[1];

// Convert rX.Y.Z to X.Y.Z
local tpuVersionFromPatchVersion(patchVersion) =
  std.lstripChars(patchVersion, 'r');

// Convert rX.Y.Z to X.Y.0
local tpuBaseVersionFromPatchVersion(patchVersion) =
  local minorVersion = minorVersionFromPatchVersion(patchVersion);
  std.lstripChars(minorVersion + '.0', 'r');

function(
  patchVersion,
  groupLabel=null,
  modes='func',
  minorVersion=minorVersionFromPatchVersion(patchVersion),
  tpuVersion=tpuVersionFromPatchVersion(patchVersion),
  tpuBaseVersion=tpuBaseVersionFromPatchVersion(patchVersion),
)
  local modesList = std.split(modes, ',');
  local tests = [
    test {
      frameworkPrefix: 'tf-%s' % patchVersion,
      image+: '-patch',
      imageTag: patchVersion,
      tpuSettings+: {
        softwareVersion: std.strReplace(test.tpuSettings.softwareVersion, tpuBaseVersion, tpuVersion),
      },
      labels+: if groupLabel != null then {
        group: groupLabel,
      } else {},

      jobTemplate+:: {
        metadata+: {
          namespace: 'automated',
        },
      },
    }
    for test in targets
    if
      std.startsWith(test.frameworkPrefix, 'tf-%s' % minorVersion) &&
      std.member(modesList, test.mode) &&
      test.accelerator.type == 'tpu'
  ];
  local clusterTests = utils.splitByCluster(
    tests, clusters.defaultCluster, clusters.acceleratorClusters,
  );
  std.foldl(
    function(result, clusterName) result + {
      ['%s/%s.yaml' % [clusterName, test.testName]]: std.manifestYamlDoc(test.oneshotJob)
      for test in clusterTests[clusterName]
    },
    std.objectFields(clusterTests),
    {},
  )
