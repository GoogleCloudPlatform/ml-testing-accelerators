# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";
local volumes = import "templates/volumes.libsonnet";

local debugJob = base.BaseTest {
  frameworkPrefix: "pt",
  modelName: "custom-job",
  mode: "bisection",
  configMaps: [],

  timeout: 3600, # 1 hour, in seconds

  image: "gcr.io/tpu-pytorch/xla_debug",
  imageTag: std.extVar('image-tag'),

  tpuSettings+: {
    softwareVersion: "pytorch-nightly",
  },
  accelerator: tpus.v3_8,

  volumeMap+: {
    datasets: volumes.PersistentVolumeSpec {
      name: "pytorch-datasets-claim",
      mountPath: "/datasets",
    },
  },
  command: utils.scriptCommand(
    |||
      export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"
      python3 pytorch/xla/test/test_train_mp_mnist.py --fake_data |& tee test_logs.txt
      global_rate=$(grep 'GlobalRate=' test_logs.txt | tail -1 | grep -oP 'GlobalRate=([0-9]*[.])?[0-9]+ ' | sed 's/[^0-9.]*//g')
      echo "Final GlobalRate was: $global_rate"
      test $(echo "$global_rate 279.99" | awk '{print ($1 > $2)}') -eq 1
    |||
  ),
};

debugJob.oneshotJob
