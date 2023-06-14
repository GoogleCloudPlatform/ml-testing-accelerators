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

local bert = import 'tf-bert-glue_mnli.libsonnet';
local dcn = import 'tf-dcn-criteo.libsonnet';
local dlrm = import 'tf-dlrm-criteo.libsonnet';
local keras_api = import 'tf-keras-api.libsonnet';
local maskrcnn = import 'tf-maskrcnn-coco.libsonnet';
local resnet = import 'tf-resnet-imagenet.libsonnet';
local resnetrs = import 'tf-resnetrs-imagenet.libsonnet';
local retinanet = import 'tf-retinanet-coco.libsonnet';
local wmt = import 'tf-wmt-wmt14_translate.libsonnet';

// Add new models here
std.flattenArrays([
  bert.configs,
  dlrm.configs,
  dcn.configs,
  keras_api.configs,
  maskrcnn.configs,
  retinanet.configs,
  resnet.configs,
  resnetrs.configs,
  wmt.configs,
])
