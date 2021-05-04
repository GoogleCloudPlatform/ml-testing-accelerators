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
local mixins = import 'templates/mixins.libsonnet';

{
  LegacyTpuTest:: common.LegacyTpuTest {
    frameworkPrefix: 'tf-r1.15.5',
    tpuSettings+: {
      softwareVersion: '1.15.5',
    },
    imageTag: 'r1.15.5',
  },
  // Running convergence tests at Midnight PST on Sunday.
  Convergence:: mixins.Convergence {
    schedule: '0 8 * * 0',
  },
  // Don't run tests manually by default since this release is stable.
  Functional:: mixins.Functional {
    schedule: null,
  },
}
