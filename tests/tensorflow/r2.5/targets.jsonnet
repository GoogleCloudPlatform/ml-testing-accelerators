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

local bert_mnli = import "bert-mnli.libsonnet";
local bert_squad = import "bert-squad.libsonnet";
local classifier_resnet = import "classifier-resnet.libsonnet";
local classifier_efficientnet = import "classifier-efficientnet.libsonnet";
local keras_api = import "keras-api.libsonnet";
local mnist = import "mnist.libsonnet";
local maskrcnn = import "maskrcnn.libsonnet";
local ncf = import "ncf.libsonnet";
local resnet_ctl = import "resnet-ctl.libsonnet";
local retinanet = import "retinanet.libsonnet";
local shapemask = import "shapemask.libsonnet";
local transformer_translate = import "transformer-translate.libsonnet";
local xlnet_imdb = import "xlnet-imdb.libsonnet";
local xlnet_squad = import "xlnet-squad.libsonnet";

# Add new models here
std.flattenArrays([
  bert_mnli.configs,
  bert_squad.configs,
  classifier_resnet.configs,
  classifier_efficientnet.configs,
  keras_api.configs,
  mnist.configs,
  maskrcnn.configs,
  ncf.configs,
  resnet_ctl.configs,
  retinanet.configs,
  shapemask.configs,
  transformer_translate.configs,
  xlnet_imdb.configs,
  xlnet_squad.configs,
])

