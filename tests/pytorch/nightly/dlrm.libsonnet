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
    git clone https://github.com/taylanbil/dlrm.git -b tpu
    pip install onnx
    python dlrm/dlrm_tpu_runner.py \
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
  local one_core = common.Functional {
    modelName: "onecore",
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
    modelName: "seq-fwd",
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
    modelName: "mp-fwd",
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
    modelName: "mpdp-fwd",
    command: utils.scriptCommand(
      |||
        %(command_common)s  --mini-batch-size=2048 \
          --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 \
          --tpu-model-parallel-group-len 4 \
          --tpu-cores=8
      ||| % command_common
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
  ]
}
