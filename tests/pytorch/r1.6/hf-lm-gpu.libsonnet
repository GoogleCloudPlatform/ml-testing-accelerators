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

local common = import "common.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local gpus = import "templates/gpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local command_common = |||
    apt-get update
    apt-get install -y git
    git clone -b v3.2.0 https://github.com/huggingface/transformers.git
    cd transformers && pip install .
    git log -1
    python -m torch.distributed.launch --nproc_per_node 4 examples/language-modeling/run_language_modeling.py \
      --logging_dir ./tensorboard-metrics \
      --cache_dir ./cache_dir \
      --train_data_file /datasets/wikitext-103-raw/wiki.train.raw \
      --do_train \
      --do_eval \
      --eval_data_file /datasets/wikitext-103-raw/wiki.valid.raw \
      --overwrite_output_dir \
      --output_dir language-modeling \
      --logging_steps 100 \
      --save_steps 3000 \
      --overwrite_cache \
  |||,
  local command_copy_metrics = |||
    gsutil -m cp -r ./tensorboard-metrics/* $(MODEL_DIR)
  |||,
  local bert_large_fp32 = common.Convergence {
    modelName: "hf-bert-large-fp32",
    command: utils.scriptCommand(
      |||
        %(common)s --mlm --model_type=bert \
        --model_name_or_path bert-large-uncased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
  },
  local bert_large_fp16 = common.Convergence {
    modelName: "hf-bert-large-fp16",
    command: utils.scriptCommand(
      |||
        %(common)s --fp16 --mlm --model_type=bert \
        --model_name_or_path bert-large-uncased \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
  },

# GPT2 OOMs on 16GB HBM V100
  #local gpt2_medium_fp32 = common.Convergence {
    #modelName: "hf-gpt2-medium-fp32",
    #command: utils.scriptCommand(
      #|||
        #%(common)s --mlm --model_type=gpt \
        #--model_name_or_path gpt2-medium \
        #--num_train_epochs 3 \
        #--per_device_train_batch_size 4 \
        #--per_device_eval_batch_size 4
        #%(common_copy)s
      #||| % {common: command_common, common_copy: command_copy_metrics}
    #),
  #},
  local gpt2_medium_fp16 = common.Convergence {
    modelName: "hf-gpt2-medium-fp16",
    command: utils.scriptCommand(
      |||
        %(common)s --fp16 --mlm --model_type=gpt2 \
        --model_name_or_path gpt2-medium \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4
        %(common_copy)s
      ||| % {common: command_common, common_copy: command_copy_metrics}
    ),
  },
 
  local hf_lm = common.PyTorchGpuTest {
    modelName: "hf-lm",
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              resources+: {
                requests: {
                  cpu: "42.0",
                  memory: "80Gi",
                },
              },
            },
          },
        },
      },
    },
  },

  local v100 = {
    accelerator: gpus.teslaV100,
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 + { count: 4 },
  },
  configs: [
    hf_lm + v100x4 + bert_large_fp32 + timeouts.Hours(24),
    hf_lm + v100x4 + bert_large_fp16 + timeouts.Hours(24),
    #hf_lm + v100x4 + gpt2_medium_fp32 + timeouts.Hours(24),
    hf_lm + v100x4 + gpt2_medium_fp16 + timeouts.Hours(24),
  ],
}
