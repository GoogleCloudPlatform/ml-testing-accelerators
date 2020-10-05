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

{
  local xlnet = common.ModelGardenTest {
    modelName: "xlnet-squad",
    command: [
      "python3",
      "official/nlp/xlnet/run_squad.py",
      "--strategy_type=tpu",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--init_checkpoint=$(XLNET_CHECKPOINT_DIR)/xlnet_model.ckpt",
      "--model_dir=$(MODEL_DIR)",
      "--train_tfrecord_path=$(XLNET_SQUAD_DIR)",
      "--test_tfrecord_path=$(XLNET_SQUAD_DIR)/12048.eval.tf_record",
      "--test_feature_path=$(XLNET_SQUAD_DIR)/spiece.model.slen-512.qlen-64.eval.features.pkl",
      "--predict_file=$(XLNET_SQUAD_DIR)/dev-v2.0.json",
      "--predict_dir=$(MODEL_DIR)",
      "--seq_len=512",
      "--reuse_len=256",
      "--mem_len=0",
      "--n_layer=24",
      "--d_model=1024",
      "--d_embed=1024",
      "--n_head=16",
      "--d_head=64",
      "--d_inner=4096",
      "--untie_r=true",
      "--ff_activation=gelu",
      "--learning_rate=.00003",
      "--warmup_steps=1000",
      "--iterations=1000",
      "--bi_data=false",
      "--query_len=64",
      "--adam_epsilon=.000001",
      "--lr_layer_decay_rate=0.75",
    ],
  },
  local functional = common.Functional {
    command+: [
      "--train_steps=1000",
    ],
  },
  local convergence = common.Convergence {
    command+: [
      "--train_steps=8000",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--train_batch_size=8",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--train_batch_size=48",
    ],
  },

  configs: [
    xlnet + v3_8 + functional,
    xlnet + v3_8 + convergence,
  ],
}
