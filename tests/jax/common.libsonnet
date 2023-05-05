// Copyright 2021 Google LLC
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
local utils = import 'templates/utils.libsonnet';

{
  JaxTest:: common.CloudAcceleratorTest + experimental.BaseTpuVmMixin {
    local config = self,

    frameworkPrefix: 'jax',
    image: 'google/cloud-sdk',
    accelerator: tpus.v2_8,

    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          exclude_tags: ['_hparams_/session_start_info'],
          merge_runs: true,
        },
        // Remove default duration assertion.
        literals+: {
          assertions+: {
            duration: null,
          },
        },
      },
    },

    jaxlibVersion:: error 'Add jaxlib version mixin',
    scriptConfig:: {
      maybeBuildJaxlib: error 'Must define `maybeBuildJaxlib`',
      installLocalJax: error 'Must define `installLocalJax`',
      installLatestJax: error 'Must define `installLatestJax`',
      testEnvWorkarounds: error 'Must define `testEnvWorkarounds`',
      printDiagnostics: |||
        python3 -c 'import jax; print("jax version:", jax.__version__)'
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
        python3 -c 'import jax; print("libtpu version:",
          jax.lib.xla_bridge.get_backend().platform_version)'
      |||,
    },

    tpuSettings+: {
      tpuVmCreateSleepSeconds: 60,
    },
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'create-tpu'+: {
            local tpuCreateSettings = {
              acceleratorName: std.escapeStringBash(config.accelerator.name),
              softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
              startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
              sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
              testName: std.strReplace(config.testName, '.', '-'),
            },
            command: utils.scriptCommand(|||
              project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              tpu_name=tpu-${POD_UID}
              ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""
              echo "
                gcloud alpha compute tpus tpu-vm delete -q ${tpu_name} --zone=${zone}
              " > /scripts/cleanup.sh

              curl -X POST \
                -H "Authorization: Bearer $(gcloud auth print-access-token)" \
                -H "Content-Type: application/json" \
                -d "{
                  accelerator_type: %(acceleratorName)s,
                  runtime_version: %(softwareVersion)s,
                  network_config: {enable_external_ips: true},
                  labels: {test_name: '%(testName)s' },
                  boot_disk: {source_image: 'projects/cloud-tpu-v2-images-dev/global/images/family/tpu-vm-base'},
                  metadata: {
                    'ssh-keys': 'xl-ml-test:$(cat /scripts/id_rsa.pub)',
                    'startup-script': %(startupScript)s,
                  }
              }" https://tpu.googleapis.com/v2alpha1/projects/${project}/locations/${zone}/nodes?node_id=${tpu_name}
              echo "Waiting for TPU Pod ${tpu_name} to become ready..."
              timeout 10m bash -c -- "
              while [[ \${health:-NONE} != READY ]];
                do sleep 60 && \
                health=\$(gcloud \
                  --project=${project} \
                  compute \
                  tpus \
                  describe \
                  ${tpu_name} \
                  --zone=${zone} \
                  --format='value(state)') && \
                echo 'Waiting for ready TPU (current state \${health:-NONE})...';
              done
              "
              echo ${zone} > /scripts/zone
              echo ${tpu_name} > /scripts/tpu_name
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" > /scripts/tpu_ip
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --flatten="networkEndpoints[]" --format="csv[no-heading](networkEndpoints.ipAddress)" > /scripts/all_tpu_ips
              sleep %(sleepTime)d
            ||| % tpuCreateSettings),
          },
        },
      },
    },

    // JAX tests are structured as bash scripts that run directly on the Cloud
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

  jaxlibHead:: {
    jaxlibVersion:: 'head',
    scriptConfig+: {
      // Install jax without jaxlib or libtpu deps
      installLocalJax: |||
        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git
        cd jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -r build/test-requirements.txt

        pip install .
      |||,
      installLatestJax: 'pip install jax',
      maybeBuildJaxlib: |||
        echo "Installing latest jaxlib-nightly..."
        pip install --pre jaxlib \
          -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
        pip list | grep jaxlib
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'

        echo "Installing latest libtpu-nightly..."
        pip install libtpu-nightly --no-index --pre \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
    },
  },

  jaxlibLatest:: {
    jaxlibVersion:: 'latest',
    scriptConfig+: {
      installLocalJax: |||
        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git
        cd jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -r build/test-requirements.txt

        pip install .[tpu] \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      installLatestJax: |||
        pip install jax[tpu] \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      maybeBuildJaxlib: '',
    },
  },

  tpuVmBaseImage:: {
    local config = self,

    tpuSettings+: {
      softwareVersion: 'tpu-vm-base',
    },
    scriptConfig+: {
      testEnvWorkarounds: |||
        pip install tensorflow
      |||,
    },
  },

  huggingFace:: {
    scriptConfig+: {
      installPackages: |||
        set -x
        set -u
        set -e

        # .bash_logout sometimes causes a spurious bad exit code, remove it.
        rm .bash_logout

        pip install --upgrade pip
        git clone https://github.com/huggingface/transformers.git
        cd transformers && pip install .
        pip install -r examples/flax/_tests_requirements.txt
        pip install --upgrade huggingface-hub urllib3 zipp

        pip install tensorflow
        pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      verifySetup: |||
        python3 -c 'import flax; print("flax version:", flax.__version__)'
        num_devices=`python3 -c "import jax; print(jax.device_count())"`
        if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
        fi
      |||,
    },
  },
}
