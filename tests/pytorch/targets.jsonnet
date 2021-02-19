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

local nightly = import "nightly/targets.jsonnet";
local r1_6 = import "r1.6/targets.jsonnet";
local r1_7 = import "r1.7/targets.jsonnet";
local r1_8 = import "r1.8/targets.jsonnet";

// Add new versions here
std.flattenArrays([
  nightly,
  r1_6,
  r1_7,
  r1_8,
])
