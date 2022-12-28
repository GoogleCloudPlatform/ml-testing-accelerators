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
local utils = import 'templates/utils.libsonnet';

{
  local command_common = |||
    pip install https://storage.googleapis.com/tpu-pytorch/wheels/torchaudio-nightly-cp37-cp37m-linux_x86_64.whl
    pip install omegaconf hydra-core soundfile
    sudo apt-get install -y libsndfile-dev
    git clone --recursive https://github.com/pytorch/fairseq.git
    pip install --editable fairseq
    export OMP_NUM_THREADS=1
    fairseq-hydra-train \
    task.data=/datasets/w2v2-librispeech-100hrs/w2v/manifest \
    optimization.max_update=%d \
    dataset.batch_size=4 \
    common.log_format=simple \
    --config-dir fairseq/examples/wav2vec/config/pretraining   \
    --config-name wav2vec2_large_librivox_tpu.yaml \
  |||,
  local w2v2 = common.PyTorchTest {
    modelName: 'w2v2',

    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    cpu: '9.0',
    memory: '30Gi',
  },
  local func = common.Functional {
    command: utils.scriptCommand(
      command_common % 500
    ),
  },
  local conv = common.Convergence {
    command: utils.scriptCommand(
      (command_common % 50000) + |||
        2>&1 | tee training_logs.txt
        loss=$(
          cat training_logs.txt | grep '| loss ' | \
          tail -1 | sed 's/.*loss //' | cut -d '|' -f1
        )
        echo 'final loss is' $loss
        test $( echo $loss | cut -d '.' -f1 ) -lt 3
      |||

    ),
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    w2v2 + v3_8 + func + timeouts.Hours(2) + mixins.Experimental,
    w2v2 + v3_8 + conv + timeouts.Hours(20) + mixins.Experimental,
  ],
}
