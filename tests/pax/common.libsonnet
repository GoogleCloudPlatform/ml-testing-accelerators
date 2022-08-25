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

local common = import '../common.libsonnet';
local experimental = import '../experimental.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  PaxTest:: common.CloudAcceleratorTest + experimental.BaseTpuVmMixin {
    local config = self,

    frameworkPrefix: 'pax',
    // TODO: set image path
    image: 'gcr.io/tbd',
    accelerator: tpus.v4_8,
    // gcs_bucket: '',
    
    tpuSettings+: {
      softwareVersion: 'tpu-vm-v4-base',
      tpuVmCreateSleepSeconds: 60,
      tpuVmPaxSetup: |||
        gsutil cp gs://pax-on-cloud-tpu-project/wheels/20220814/paxml-nightly+20220814-py3-none-any.whl .
        pip install paxml-nightly+20220814-py3-none-any.whl
        pip install praxis
      |||,
    },    


    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          exclude_tags: ['_hparams_/session_start_info'],
          merge_runs: true,
        },
        // // Remove default duration assertion.
        // literals+: {
        //   assertions+: {
        //     duration: null,
        //   },
        // },
      },
    },

    // scriptConfig+: {
    //   testEnvWorkarounds: |||
    //     pip install tensorflow
    //   |||,
    // },
  },
}
