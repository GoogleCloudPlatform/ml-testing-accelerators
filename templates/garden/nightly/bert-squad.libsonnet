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
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local bert = base.GardenTest {
    modelName: "bert-squad",
    command: [
      "python3",
      "official/nlp/bert/run_squad.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--input_meta_data_path=gs://xl-ml-test-us-central1/data/squad/squad_v1.1_meta_data.json",
      "--train_data_path=gs://xl-ml-test-us-central1/data/squad/squad_v1.1_train.tfrecord",
      "--predict_file=gs://xl-ml-test-us-central1/data/squad/dev-v1.1.json",
      "--vocab_file=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/vocab.txt",
      "--bert_config_file=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_config.json",
      "--init_checkpoint=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_model.ckpt",
      "--learning_rate=8e-5",
      "--do_lower_case=true",
      "--distribution_strategy=tpu",
      "--steps_per_loop=500",
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--mode=train",
      "--num_train_epochs=1",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--train_batch_size=16",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--train_batch_size=32",
    ],
  },

  configs: [
    bert + v2_8 + functional,
    bert + v3_8 + functional,
  ],
}
