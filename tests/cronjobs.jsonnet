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

local clusters = import 'clusters.jsonnet';
local utils = import 'templates/utils.libsonnet';

local all_tests = import 'all_tests.jsonnet';

local copyrightHeader = |||
  # Copyright 2020 Google LLC
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  #     https://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

|||;

local cronJobs = utils.cronJobOutput(
  all_tests,
  clusters.defaultCluster,
  clusters.acceleratorClusters,
);

// Outputs {filename: yaml_string} for each target
{
  [filename]: copyrightHeader + cronJobs[filename]
  for filename in std.objectFields(cronJobs)
}
