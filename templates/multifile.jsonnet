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

local targets = import "all_targets.jsonnet";

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

local defaultRegion = "us-central1";
local getDirectory(target) = (if target.accelerator.region == null then defaultRegion else target.accelerator.region) + "/" + "gen/";

# Outputs {filename: yaml_string} for each target
{
  [getDirectory(targets[name]) + name + ".yaml"]: copyrightHeader + std.manifestYamlDoc(
    targets[name].cronJob
  ) for name in std.objectFields(targets) if targets[name].schedule != null
}
