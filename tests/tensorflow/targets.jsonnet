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

local nightly = import 'nightly/targets.jsonnet';
local r2_10 = import 'r2.10/targets.jsonnet';
local r2_11 = import 'r2.11/targets.jsonnet';
local r2_7 = import 'r2.7/targets.jsonnet';
local r2_8 = import 'r2.8/targets.jsonnet';
local r2_9 = import 'r2.9/targets.jsonnet';

// Add new versions here
std.flattenArrays([
  nightly,
  r2_11,
])
