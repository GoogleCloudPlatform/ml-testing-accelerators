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
  local dlrm = {
    local config = self,

    modelName: 'dlrm',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    paramsOverride:: {
      scriptPath: 'tpu-examples/deps/dlrm/dlrm_tpu_runner.py',
      miniBatchSize: 256,
      archEmbeddingSize: 1000000-1000000,
      tpuModelParallelGroupLen: 1,
      tpuCores: 1,
      ArchSparseFeatureSize: 64,
      ArchMlpBot: "512-512-64",
      ArchAlpTop: "1024-1024-1024-1",
      NumIndicesPerLookup: 100,
      trainCommand: [
        'python3',
        self.scriptPath,
        '--arch-interaction-op=dot',
        '--lr-num-warmup-steps=10',
        '--lr-decay-start-step=10',
        '--num-batches=1000',
        '--data-generation="random"',
        '--numpy-rand-seed=72',
        '--print-freq=100',
        '--use-tpu',
        '--metrics-debug',
        '--num-indices-per-lookup-fixed',
        '--mini-batch-size=%d' % config.paramsOverride.miniBatchSize,
        '--arch-embedding-size=%d' % config.paramsOverride.archEmbeddingSize,
        '--tpu-model-parallel-group-len=%d' % config.paramsOverride.tpuModelParallelGroupLen,
        '--tpu-cores=%d' % config.paramsOverride.tpuCores,
      ],
    },
    cpu: '9.0',
    memory: '30Gi',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap:: {},
        },
      },
    },
  },
  
  local dlrm_convergence = common.PyTorchTest {
    modelName: 'dlrm-convergence',

    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              requests: {
                cpu: '40.0',
                memory: '500Gi',
              },
            },
          },
        },
      },
    },
  },
  local one_core = common.Functional {
    modelName: 'dlrm-onecore',
    paramsOverride+:: {
      miniBatchSize: 256,
      archEmbeddingSize: 1000000-1000000,
      tpuModelParallelGroupLen: 1,
      tpuCores: 1,
    },
    command: utils.scriptCommand(
      |||
        pip install onnx
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        %s 
      ||| % utils.toCommandString(self.paramsOverride.trainCommand),
    ),
  },
  local seq_fwd = common.Functional {
    modelName: 'dlrm-seq-fwd',
    paramsOverride+:: {
      miniBatchSize: 2048,
      archEmbeddingSize: 1000000-1000000,
      tpuModelParallelGroupLen: 1,
      tpuCores: 8,
    },
    command: utils.scriptCommand(
      |||
        pip install onnx
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        %s 
      ||| % utils.toCommandString(self.paramsOverride.trainCommand),
    ),
  },
  local mp_fwd = common.Functional {
    modelName: 'dlrm-mp-fwd',
    paramsOverride+:: {
      miniBatchSize: 2048,
      archEmbeddingSize: 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000,
      tpuModelParallelGroupLen: 8,
      tpuCores: 8,
    },
    command: utils.scriptCommand(
      |||
        pip install onnx
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        %s 
      ||| % utils.toCommandString(self.paramsOverride.trainCommand),
    ),
  },
  local mp_dp_fwd = common.Functional {
    modelName: 'dlrm-mpdp-fwd',
    paramsOverride+:: {
      miniBatchSize: 2048,
      archEmbeddingSize: 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000,
      tpuModelParallelGroupLen: 4,
      tpuCores: 8,
    },
    command: utils.scriptCommand(
      |||
        pip install onnx
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        %s 
      ||| % utils.toCommandString(self.paramsOverride.trainCommand),
    ),
  },

  local criteo_kaggle = common.Convergence {
    paramsOverride+:: {
      tpuCores: 8,
      miniBatchSize: 128,
      tpuModelParallelGroupLen: 8,
      ArchSparseFeatureSize: 16,
      ArchMlpBot: "13-512-256-64-16",
      ArchAlpTop: "512-256-1",
      NumIndicesPerLookup: 1,
      trainCommand+: [
        '--raw-data-file=/datasets/criteo-kaggle-mm/train.txt',
        '--processed-data-file=/datasets/criteo-kaggle-mm/kaggleAdDisplayChallenge_processed.npz',
        '--memory-map',
        '--data-generation=dataset',
        '--print-time',
        '--test-mini-batch-size=16384',
        '--test-freq=101376',
        '--data-set=kaggle',
        '--loss-function=bce',
        '--round-targets=True',
        '--learning-rate=0.1',
        '--print-freq=1024',
        '--no-save',
        '--max-epoch=1',
      ],
    },
    command: utils.scriptCommand(
      |||
        set +e
        apt-get install -y bc
        pip install onnx
        git clone --recursive https://github.com/pytorch-tpu/examples.git
        
        %s 
        --raw-data-file=/datasets/criteo-kaggle-mm/train.txt \
        --processed-data-file=/datasets/criteo-kaggle-mm/kaggleAdDisplayChallenge_processed.npz \
        --memory-map \
        |& tee dlrm_logs.txt
        acc=`grep Testing dlrm_logs.txt | tail -1 | grep -oP 'best \K[+-]?([0-9]*[.])?[0-9]+'`
        echo 'Accuracy is' $acc
        test $(echo $acc'>'78.75 | bc -l) -eq 1  # assert cls acc higher than 78.75
      ||| % [
        utils.toCommandString(self.paramsOverride.trainCommand),
      ]
    ),
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    common.PyTorchTest + dlrm + v3_8 + one_core + timeouts.Hours(3),
    common.PyTorchTest + dlrm + v3_8 + seq_fwd + timeouts.Hours(3),
    common.PyTorchTest + dlrm + v3_8 + mp_fwd + timeouts.Hours(3),
    common.PyTorchTest + dlrm + v3_8 + mp_dp_fwd + timeouts.Hours(3),
    common.PyTorchTest + dlrm + v3_8 + criteo_kaggle + timeouts.Hours(6),
  ],
}
