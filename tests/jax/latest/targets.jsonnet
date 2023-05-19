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
// limitations under the License,

local hf_bart = import 'flax-bart-wiki_summary.libsonnet';
local hf_bert_mnli = import 'flax-bert-glue_mnli.libsonnet';
local hf_bert_mrpc = import 'flax-bert-glue_mrpc.libsonnet';
local hf_gpt2 = import 'flax-gpt2-oscar.libsonnet';
local resnet = import 'flax-resnet-imagenet.libsonnet';
local hf_sd = import 'flax-sd-pokemon.libsonnet';
local hf_vit = import 'flax-vit-imagenette.libsonnet';
local wmt = import 'flax-wmt-wmt17_translate.libsonnet';

// Add new models here
std.flattenArrays([
  hf_bart.configs,
  hf_vit.configs,
  hf_bert_mnli.configs,
  hf_bert_mrpc.configs,
  hf_gpt2.configs,
  resnet.configs,
  wmt.configs,
  hf_sd.configs,
])
