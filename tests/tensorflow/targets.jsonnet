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
local r1_15 = import 'r1.15/targets.jsonnet';
local r2_1 = import 'r2.1/targets.jsonnet';
local r2_2 = import 'r2.2/targets.jsonnet';
local r2_3 = import 'r2.3/targets.jsonnet';
local r2_4 = import 'r2.4/targets.jsonnet';
local r2_5 = import 'r2.5/targets.jsonnet';
local r2_6 = import 'r2.6/targets.jsonnet';
local r2_7 = import 'r2.7/targets.jsonnet';

// Add new versions here
std.flattenArrays([
  nightly,
  r1_15,
  r2_1,
  r2_2,
  r2_3,
  r2_4,
  r2_5,
  r2_6,
  r2_7,
])
