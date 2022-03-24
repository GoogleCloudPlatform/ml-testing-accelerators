// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  TensorFlowTpuVmMixin:: experimental.BaseTpuVmMixin + mixins.Unsuspended {
    local config = self,
    tpuSettings+: {
      tpuVmEnvVars+: {
        PYTHONPATH: '${PWD}',
      } + if config.accelerator.replicas > 1 then {
        TPU_LOAD_LIBRARY: '0',
      } else {},

      softwareVersion+: if config.accelerator.replicas > 1 then
        '-pod'
      else
        '',
    },

    podTemplate+:: {
      spec+: {
        containerMap+:: {
          train+: {
            args: null,
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'pip install -r /usr/share/tpu/models/official/requirements.txt'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'pip install tensorflow-recommenders --no-deps'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'cd /usr/share/tpu/models; %(env)s '%(testCommand)s
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % {
                testCommand: std.escapeStringBash(
                  std.join(
                    ' ',
                    ['"' + std.strReplace(c, '"', '\\"') + '"' for c in config.command],
                  ),
                ),
                env: std.join(
                  ' ',
                  [
                    '%s=%s' % [key, config.tpuSettings.tpuVmEnvVars[key]]
                    for key in std.objectFields(config.tpuSettings.tpuVmEnvVars)
                  ],
                ),
              },
            ],
          },
        },
      },
    },
  },
  TensorFlowTpuVmDockerTest:: experimental.BaseTpuVmMixin {
    local config = self,
    tpuSettings+: {
      tpuVmDockerArgs: if config.accelerator.replicas == 1 then
        '-v "/lib/libtpu.so:/lib/libtpu.so"'
      else
        '-v "/lib/libtpu.so:/lib/libtpu.so" --net host -e TPU_LOAD_LIBRARY=0',
    },

    image: 'gcr.io/xl-ml-test/tensorflow-1vm:nightly',

    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local remoteScript = {
              dockerImage: config.image,
              dockerArgs: config.tpuSettings.tpuVmDockerArgs,
              dockerCommand: std.escapeStringBash(
                std.join(
                  ' ',
                  ['"' + std.strReplace(c, '"', '\\"') + '"' for c in config.command],
                ),
              ),
            },
            args: null,
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo usermod -aG docker $USER'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'gcloud auth configure-docker'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo gcsfuse --implicit-dirs -o allow_other /gcs'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'docker run -i --rm --privileged -v "/gcs:/gcs" -v "$(LOCAL_OUTPUT_DIR):$(LOCAL_OUTPUT_DIR)" --entrypoint "" %(dockerArgs)s %(dockerImage)s '%(dockerCommand)s
                exit_code=$?
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) 'gsutil -m cp -r $(LOCAL_OUTPUT_DIR) $(MODEL_DIR)'
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % remoteScript,
            ],
          },
        },
      },
    },
  },
  TensorflowServingTpuVmMixin:: experimental.BaseTpuVmTest {
    local config = self,
    local image = error 'must supply base `image`.',
    tpuSettings+: {
      tpuVmStartupScript: 'gcloud auth configure-docker && ' +
                          'mkdir -p /models/%(model)s && ' % config.servingConfig +
                          'gsutil -m cp -R %(gcsDir)s/* /models/%(model)s && ' % config.servingConfig +
                          'docker run -d --privileged -e MODEL_NAME=%(model)s -e TPU_MIN_LOG_LEVEL=0 -p 8500:8500 -v "/models:/models" -v "/lib/libtpu.so:/lib/libtpu.so" %(modelServerImage)s' % config.servingConfig,
      tpuVmCreateSleepSeconds: 120,
    },

    podTemplate+: {
      spec+: {
        containerMap+:: {
          train+: {
            local scriptSettings = {
              testCommand:
                std.join(
                  ' ',
                  config.command,
                ),
            },
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u

                %(testCommand)s
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },
}
