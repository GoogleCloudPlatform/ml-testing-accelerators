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

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local gpus = import 'templates/gpus.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local servingTest = common.ServingTest {
    local config = self,
    command: [
      'python3',
      'tpu/models/experimental/inference/load_test/examples/loadgen_grpc_main.py',
      '--target=grpc://$(cat /scripts/tpu_ip):8500',
      '--model_name=%(model)s' % config,
      '--scenario=server',
      '--performance_sample_count=100',
      '--data_type=%(dataType)s' % config,
      '--target_latency_percentile=0.99',
      '--target_latency_ns=100000000',
      '--batch_size=%(batchSize)d' % config,
      '--duration_ms=1000',
      '--query_count=100',
    ],
  },
  local functional = common.Functional,
  local bert = servingTest {
    model: 'bert',
    dataType: 'synthetic_bert',
    batchSize: 4,
    //gcsDir: '$(SERVING_MODELS_DIR)/bert-base-tf2/tpu',
    gcsDir: 'gs://serving-benchmarks/bert-base-tf2/tpu',
  },
  local tpuVm = experimental.TensorflowServingTpuVmMixin,

  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    bert + functional + v2_8 + tpuVm,
    bert + functional + v3_8 + tpuVm,
  ],
}
