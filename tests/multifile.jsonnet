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

local defaultRegion = 'us-central1';
local regionAccelerators = {
  'us-central1': [],
  'europe-west4': [
    tpus.v2_32,
    tpus.v3_32,
  ],
};

local cronJobs = utils.cronJobOutput(all_tests, defaultRegion, regionAccelerators);

// Outputs {filename: yaml_string} for each target
{
  [filename]: copyrightHeader + cronJobs[filename]
  for filename in std.objectFields(cronJobs)
}
