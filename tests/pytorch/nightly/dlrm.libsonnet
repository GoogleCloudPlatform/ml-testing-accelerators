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
local tpus = import "templates/tpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local command_common = |||
    pip install onnx
    git clone --recursive https://github.com/pytorch-tpu/examples.git
    python examples/deps/dlrm/dlrm_tpu_runner.py \
      --arch-sparse-feature-size=64 \
      --arch-mlp-bot=512-512-64 \
      --arch-mlp-top=1024-1024-1024-1 \
      --arch-interaction-op=dot \
      --lr-num-warmup-steps 10 \
      --lr-decay-start-step 10 \
      --num-batches=1000 \
      --data-generation="random" \
      --numpy-rand-seed=727 \
      --print-freq 100 \
      --num-indices-per-lookup=100 \
      --use-tpu \
      --metrics-debug \
      --num-indices-per-lookup-fixed \
  |||,
  local dlrm = common.PyTorchTest {
    modelName: "dlrm",
    schedule: "0 21 * * *",
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
                  cpu: "9.0",
                  memory: "30Gi",
                },
              },
            },
          },
        },
      },
    },
  },
  local dlrm_convergence = common.PyTorchTest {
    modelName: "dlrm-convergence",
    schedule: "0 21 * * *",
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
                  cpu: "40.0",
                  memory: "500Gi",
                },
              },
            },
          },
        },
      },
    },
  },
  local one_core = common.Functional {
    modelName: "dlrm-onecore",
    command: utils.scriptCommand(
      |||
        %(command_common)s  --mini-batch-size=256 \
          --arch-embedding-size=1000000-1000000 \
          --tpu-model-parallel-group-len 1 \
          --tpu-cores=1
      ||| % command_common
    ),
  },
  local seq_fwd = common.Functional {
    modelName: "dlrm-seq-fwd",
    command: utils.scriptCommand(
      |||
        %(command_common)s  --mini-batch-size=2048 \
          --arch-embedding-size=1000000-1000000 \
          --tpu-model-parallel-group-len 1 \
          --tpu-cores=8
      ||| % command_common
    ),
  },
  local mp_fwd = common.Functional {
    modelName: "dlrm-mp-fwd",
    command: utils.scriptCommand(
      |||
        %(command_common)s  --mini-batch-size=2048 \
          --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
          --tpu-model-parallel-group-len 8 \
          --tpu-cores=8
      ||| % command_common
    ),
  },
  local mp_dp_fwd = common.Functional {
    modelName: "dlrm-mpdp-fwd",
    command: utils.scriptCommand(
      |||
        %(command_common)s  --mini-batch-size=2048 \
          --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
          --tpu-model-parallel-group-len 4 \
          --tpu-cores=8
      ||| % command_common
    ),
  },
  local criteo_kaggle = common.Convergence {
    command: utils.scriptCommand(
      |||
        apt-get install -y bc
        pip install onnx
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        python examples/deps/dlrm/dlrm_tpu_runner.py \
            --arch-sparse-feature-size=16 \
            --arch-mlp-bot="13-512-256-64-16" \
            --arch-mlp-top="512-256-1" \
            --data-generation=dataset \
            --data-set=kaggle \
            --raw-data-file=/datasets/criteo-kaggle/train.txt \
            --processed-data-file=/datasets/criteo-kaggle/kaggleAdDisplayChallenge_processed.npz \
            --loss-function=bce \
            --round-targets=True \
            --learning-rate=0.1 \
            --mini-batch-size=128 \
            --print-freq=1024 \
            --print-time \
            --test-mini-batch-size=16384 \
            --test-num-workers=4 \
            --test-freq=101376 \
                --use-tpu \
                --num-indices-per-lookup=1 \
                --num-indices-per-lookup-fixed \
                --tpu-model-parallel-group-len 8 \
                --tpu-metrics-debug \
                --tpu-cores=8 |& tee dlrm_logs.txt
        acc=`grep Testing dlrm_logs.txt | tail -1 | grep -oP 'best \K[+-]?([0-9]*[.])?[0-9]+'`
        echo 'Accuracy is' $acc
        test $(echo $acc'>'78.75 | bc -l) -eq 1  # assert cls acc higher than 78.75
      |||
    ),
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    dlrm + v3_8 + one_core + timeouts.Hours(3),
    dlrm + v3_8 + seq_fwd + timeouts.Hours(3),
    dlrm + v3_8 + mp_fwd + timeouts.Hours(3),
    dlrm + v3_8 + mp_dp_fwd + timeouts.Hours(3),
    dlrm_convergence + v3_8 + criteo_kaggle + timeouts.Hours(6),
  ]
}
