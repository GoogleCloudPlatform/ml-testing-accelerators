# Copyright 2021 Google LLC
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

local common = import "../common.libsonnet";
local experimental = import "../experimental.libsonnet";

{
  JaxTest:: common.CloudAcceleratorTest + experimental.TpuVmBaseTest {
    local config = self,

    frameworkPrefix: 'jax',
    image: 'google/cloud-sdk',

    tpuSettings+: {
      softwareVersion: error "Must define `tpuSettings.softwareVersion`",
      tpuVmCreateSleepSeconds: 60,
    },

    // JAX tests are structured as bash scripts that run directly on the Cloud
    // TPU VM instead of using docker images
    testScript:: error "Must define `testScript`",
    command: [
      "bash",
      "-c",
      |||
        set -x
        set -u
        ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) << 'EOF'
          %s
        EOF
        exit_code=$?
        bash /scripts/cleanup.sh
        exit $exit_code
      ||| % config.testScript,
    ],
  },
}
