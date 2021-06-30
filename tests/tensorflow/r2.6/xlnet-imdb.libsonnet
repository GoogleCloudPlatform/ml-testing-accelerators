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

local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local xlnet = common.ModelGardenTest {
    modelName: 'xlnet-imdb',
    command: [
      'python3',
      'official/nlp/xlnet/run_classifier.py',
      '--strategy_type=tpu',
      '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
      '--train_tfrecord_path=$(XLNET_IMDB_DIR)/spiece.model.len-512.train.tf_record',
      '--test_tfrecord_path=$(XLNET_IMDB_DIR)/spiece.model.len-512.dev.eval.tf_record',
      '--init_checkpoint=$(XLNET_CHECKPOINT_DIR)/xlnet_model.ckpt',
      '--model_dir=$(MODEL_DIR)',
      '--test_data_size=25024',
      '--seq_len=512',
      '--n_layer=24',
      '--d_model=1024',
      '--d_embed=1024',
      '--n_head=16',
      '--d_head=64',
      '--d_inner=4096',
      '--untie_r=true',
      '--n_class=2',
      '--ff_activation=gelu',
      '--learning_rate=2e-5',
      '--warmup_steps=500',
      '--iterations=500',
      '--bi_data=false',
      '--summary_type=last',
    ],
  },
  local functional = common.Functional {
    command+: [
      '--train_steps=1000',
    ],
  },
  local convergence = common.Convergence {
    command+: [
      '--train_steps=4000',
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      '--train_batch_size=16',
      '--test_batch_size=16',
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      '--train_batch_size=32',
      '--test_batch_size=32',
    ],
  },
  local v2_32 = {
    accelerator: tpus.v2_32,
    command+: [
      '--train_batch_size=64',
      '--test_batch_size=64',
    ],
  },
  local v3_32 = {
    accelerator: tpus.v3_32,
    command+: [
      '--train_batch_size=128',
      '--test_batch_size=128',
    ],
  },

  configs: [
    xlnet + v3_8 + functional + mixins.Unsuspended,
    xlnet + v2_8 + convergence + timeouts.Hours(2),
    xlnet + v3_8 + convergence + timeouts.Hours(1),
    xlnet + v3_32 + convergence,
  ],
}
