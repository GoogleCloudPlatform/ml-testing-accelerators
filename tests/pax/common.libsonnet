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
    accelerator: tpus.v4_8,
    tpuSettings+: {
      softwareVersion: 'tpu-vm-v4-base',
      tpuVmCreateSleepSeconds: 60,
    },    

    // PAX tests are structured as bash scripts that run directly on the Cloud
    // TPU VM instead of using docker images
    testScript:: error 'Must define `testScript`',
    command: [
      'bash',
      '-c',
      |||
        set -x
        set -u

        cat > testsetup.sh << 'TEST_SCRIPT_EOF'
        %s
        TEST_SCRIPT_EOF

        gcloud alpha compute tpus tpu-vm ssh xl-ml-test@$(cat /scripts/tpu_name) \
        --zone=$(cat /scripts/zone) \
        --ssh-key-file=/scripts/id_rsa \
        --strict-host-key-checking=no \
        --internal-ip \
        --worker=all \
        --command "$(cat testsetup.sh)"

        exit_code=$?
        bash /scripts/cleanup.sh
        exit $exit_code
      ||| % config.testScript,
    ],
  },
}
