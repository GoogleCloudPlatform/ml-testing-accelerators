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
local mixins = import "templates/mixins.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local gpus = import "templates/gpus.libsonnet";

{
  local bert = common.ModelGardenTest {
    modelName: "bert-squad",
    command: [
      "python3",
      "official/nlp/bert/run_squad.py",
      "--input_meta_data_path=gs://xl-ml-test-us-central1/data/squad/squad_v1.1_meta_data.json",
      "--train_data_path=gs://xl-ml-test-us-central1/data/squad/squad_v1.1_train.tfrecord",
      "--predict_file=gs://xl-ml-test-us-central1/data/squad/dev-v1.1.json",
      "--learning_rate=8e-5",
      "--do_lower_case=true",
      "--model_dir=$(MODEL_DIR)",
    ],
  },
  local functional = common.Functional {
    command+: [
      "--mode=train",
      "--num_train_epochs=1",
    ],
  },
  local convergence = common.Convergence {
    command+: [
      "--num_train_epochs=2",
    ],
  },

  local gpu_common = {
    local config = self,

    command+: [
      "--vocab_file=$(KERAS_BERT_DIR)/uncased_L-12_H-768_A-12/vocab.txt",
      "--bert_config_file=$(KERAS_BERT_DIR)/uncased_L-12_H-768_A-12/bert_config.json",
      "--init_checkpoint=$(KERAS_BERT_DIR)/uncased_L-12_H-768_A-12/bert_model.ckpt",
      "--num_gpus=%d" % config.accelerator.count,
    ],
  },
  local k80 = gpu_common {
    local config = self,

    accelerator: gpus.teslaK80,
    command+: [
      "--train_batch_size=%d" % (8 * config.accelerator.replicas),
      "--predict_batch_size=%d" % (8 * config.accelerator.replicas),
    ],
  },
  local k80x8 = k80 {
    accelerator: gpus.teslaK80 + { count: 8 },
    command+: [
      "--all_reduce_alg=hierarchical_copy",
    ],
  },
  local v100 = gpu_common {
    local config = self,

    accelerator: gpus.teslaV100,
    command+: [
      "--train_batch_size=%d" % (8 * config.accelerator.replicas),
      "--predict_batch_size=%d" % (8 * config.accelerator.replicas),
    ],
  },
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 + { count: 4 },
  },

  local tpu_common = {
    command+: [
      "--vocab_file=$(KERAS_BERT_DIR)/uncased_L-24_H-1024_A-16/vocab.txt",
      "--bert_config_file=$(KERAS_BERT_DIR)/uncased_L-24_H-1024_A-16/bert_config.json",
      "--init_checkpoint=$(KERAS_BERT_DIR)/uncased_L-24_H-1024_A-16/bert_model.ckpt",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--distribution_strategy=tpu",
      "--steps_per_loop=500",
    ],
  },
  local v2_8 = tpu_common {
    accelerator: tpus.v2_8,
    command+: [
      "--train_batch_size=16",
    ],
  },
  local v3_8 =  tpu_common {
    accelerator: tpus.v3_8,
    command+: [
      "--train_batch_size=32",
    ],
  },

  configs: [
    bert + k80 + functional + timeouts.Hours(6) + mixins.Experimental,
    bert + k80 + convergence + timeouts.Hours(12) + mixins.Experimental,
    bert + k80x8 + functional + timeouts.Hours(2),
    bert + k80x8 + convergence + timeouts.Hours(4),
    bert + v100 + functional + timeouts.Hours(3),
    bert + v100 + convergence + timeouts.Hours(6),
    bert + v100x4 + functional + timeouts.Hours(2),
    bert + v100x4 + convergence + timeouts.Hours(4),
    bert + v2_8 + functional,
    bert + v3_8 + functional,
  ],
}
