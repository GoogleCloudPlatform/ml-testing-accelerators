// Copyright 2022 Google LLC
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
// limitations under the License,

local c4spmd1b_pretraining = import 'c4spmd1b.libsonnet';
local lmcloudspmdadam = import 'lmcloudspmdadam.libsonnet';
local spmd = import 'lmspmd2b.libsonnet';
local transformer = import 'lmtransformeradam.libsonnet';

// Add new models here
std.flattenArrays([
  c4spmd1b_pretraining.configs,
  spmd.configs,
  transformer.configs,
  lmcloudspmdadam.configs,
])
