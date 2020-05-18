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

local base = import "base.libsonnet";
local tpus = import "templates/tpus.libsonnet";

{
  local bert = base.LegacyTpuTest {
    modelName: "bert",
    command: [
      "python3",
      "/bert/run_classifier.py",
      "--tpu_name=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--iterations_per_loop=100",
      "--mode=train",
      "--task_name=MNLI",
      "--do_train=true",
      "--data_dir=$(TF1_BERT_DIR)/glue_data/MNLI",
      "--vocab_file=$(TF1_BERT_DIR)/2018_10_18/uncased_L-12_H-768_A-12/vocab.txt",
      "--bert_config_file=$(TF1_BERT_DIR)/2018_10_18/uncased_L-12_H-768_A-12/bert_config.json",
      "--init_checkpoint=$(TF1_BERT_DIR)/2018_10_18/uncased_L-12_H-768_A-12/bert_model.ckpt",
      "--learning_rate=3e-5",
      "--num_train_epochs=3",
      "--max_seq_length=128",
      "--use_tpu=True",
      "--output_dir=$(MODEL_DIR)",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--train_batch_size=128",
      "--num_tpu_cores=8",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--train_batch_size=128",
      "--num_tpu_cores=8",
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      "--train_batch_size=512",
      "--num_tpu_cores=32",
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      "--train_batch_size=512",
      "--num_tpu_cores=32",
    ],
  },
  local convergence = base.Convergence,

  configs: [
    bert + v2_8 + convergence,
    bert + v3_8 + convergence,
    bert + v2_32 + convergence,
    bert + v3_32 + convergence,
  ],
}
