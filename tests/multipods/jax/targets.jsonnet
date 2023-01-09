//
//Copyright 2021 Google LLC
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

local mxlaUnitTests = import 'mxla-unit-tests.libsonnet';
local syncDevicesTests = import 'sync-devices.libsonnet';
local topologyDiscoveryTests = import 'topology-discovery-tests.libsonnet';
local jaxTestrun = import 'jaxTestrun.libsonnet';
local gpt1like_slice1_v416 = import 'gpt1like-slice=1-v416.libsonnet';
local gpt1like_slice2_v416 = import 'gpt1like-slice=2-v416.libsonnet';
local gpt1like_slice4_v416 = import 'gpt1like-slice=4-v416.libsonnet';
local orbaxTests = import 'orbaxCheckpointTestrun.libsonnet';
local orbaxAsyncTests = import 'orbaxAsyncCheckpointTestrun.libsonnet';

std.flattenArrays([
  topologyDiscoveryTests.configs,
  mxlaUnitTests.configs,
  syncDevicesTests.configs,
  jaxTestrun.configs,
  gpt1like_slice1_v416.configs,
  gpt1like_slice2_v416.configs,
  gpt1like_slice4_v416.configs,
  orbaxTests.configs,
  orbaxAsyncTests.configs,
])
