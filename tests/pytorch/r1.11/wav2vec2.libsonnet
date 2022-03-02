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
// limitations under the License.

local common = import 'common.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local command_common = |||
    pip install omegaconf hydra-core soundfile
    sudo apt-get install -y libsndfile-dev
    git clone --recursive https://github.com/pytorch/fairseq.git
    pip install --editable fairseq
    export OMP_NUM_THREADS=1
    python fairseq/train.py \
       /datasets/w2v2-librispeech-100hrs/w2v/manifest/ \
       --num-batch-buckets 3 \
       --tpu \
       --max-sentences 4 \
       --max-sentences-valid 4 \
       --required-batch-size-multiple 4 \
       --distributed-world-size 8 \
       --distributed-port 12597 \
       --update-freq 1 \
       --enable-padding \
       --log-interval 20 \
       --num-workers 6 \
       --task audio_pretraining \
       --criterion wav2vec \
       --arch wav2vec2 \
       --log-keys  "['prob_perplexity','code_perplexity','temp']" \
       --quantize-targets \
       --extractor-mode default \
       --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' \
       --final-dim 256 \
       --latent-vars 320 \
       --latent-groups 2 \
       --latent-temp '(2,0.5,0.999995)' \
       --infonce \
       --optimizer adam \
       --adam-betas '(0.9,0.98)' \
       --adam-eps 1e-06 \
       --lr-scheduler polynomial_decay \
       --total-num-update 400000 \
       --lr 0.0005 \
       --warmup-updates 32000 \
       --mask-length 10 \
       --mask-prob 0.65 \
       --mask-selection static \
       --mask-other 0 \
       --mask-channel-prob 0.1 \
       --encoder-layerdrop 0 \
       --dropout-input 0.0 \
       --dropout-features 0.0 \
       --feature-grad-mult 0.1 \
       --loss-weights '[0.1, 10]' \
       --conv-pos 128 \
       --conv-pos-groups 16 \
       --num-negatives 100 \
       --cross-sample-negatives 0 \
       --max-sample-size 250000 \
       --min-sample-size 32000 \
       --dropout 0.0 \
       --attention-dropout 0.0 \
       --weight-decay 0.01 \
       --max-tokens 1400000 \
       --skip-invalid-size-inputs-valid-test \
       --ddp-backend no_c10d \
       --log-format simple \
  |||,
  local w2v2 = common.PyTorchTest {
    modelName: 'w2v2',

    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '40.0',
    memory: '300Gi',
  },
  local func = common.Functional {
    command: utils.scriptCommand(
      |||
        %(command_common)s  --max-update 500
      ||| % command_common
    ),
  },
  local conv = common.Convergence {
    command: utils.scriptCommand(
      |||
        %(command_common)s  --max-update 50000 \
          2>&1 | tee training_logs.txt
        loss=$(
          cat training_logs.txt | grep '| loss ' | \
          tail -1 | sed 's/.*loss //' | cut -d '|' -f1
        )
        echo 'final loss is' $loss
        test $( echo $loss | cut -d '.' -f1 ) -lt 3
      ||| % command_common
    ),
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    w2v2 + v3_8 + func + timeouts.Hours(2),
    w2v2 + v3_8 + conv + timeouts.Hours(20),
  ],
}
