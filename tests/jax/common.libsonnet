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

    // JAX tests are structured as bash scripts that run directly on the Cloud
    // TPU VM instead of using docker images
    testScript:: error 'Must define `testScript`',
    command: [
      'bash',
      '-c',
      |||
        set -x
        set -u
        ssh -i scripts/id_rsa -o StrictHostKeyChecking=no \
            xl-ml-test@$(cat /scripts/tpu_ip) << 'TEST_SCRIPT_EOF'
          %s
        TEST_SCRIPT_EOF
        exit_code=$?
        bash /scripts/cleanup.sh
        exit $exit_code
      ||| % config.testScript,
    ],
  },

  JaxPodTest:: self.JaxTest {
    local config = self,

    accelerator: tpus.v2_32,

    // Execute testScript on every host in the pod slice.
    command: [
      'bash',
      '-c',
      |||
        set -u
        # Asynchronously run testScript on each host via ssh, log each host's
        # output, and collect process IDs.
        pids=()
        for tpu_ip in $(cat /scripts/all_tpu_ips)
        do
          echo "Starting test script on TPU host $tpu_ip..."
          ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$tpu_ip \
              > /tmp/$tpu_ip.log 2>&1 <<'TEST_SCRIPT_EOF' &
            %s
        TEST_SCRIPT_EOF
          pids+=( $! )
          echo "pid: ${pids[-1]}"
        done

        # Wait for each host's ssh process to complete and collect exit codes.
        # We'll return an error if any process failed.
        exit_code=0
        for pid in ${pids[@]}
        do
          echo "Waiting for pid $pid to complete..."
          wait $pid
          pid_exit_code=$?
          echo "exit code: $pid_exit_code"
          if [ "$pid_exit_code" -ne "0" ]
          then
            exit_code=$pid_exit_code
          fi
        done

        # Output each host's log so it shows up in the GKE logs.
        for tpu_ip in $(cat /scripts/all_tpu_ips)
        do
          echo "========== output for TPU host $tpu_ip =========="
          cat /tmp/$tpu_ip.log
          echo "========== end of output for TPU host $tpu_ip =========="
        done

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
        cd ~/jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -r build/test-requirements.txt

        pip install .
      |||,
      installLatestJax: 'pip install jax',
      maybeBuildJaxlib: |||
        echo "Building jaxlib from source at TF head"
        echo "Checking out TF..."
        cd ~/
        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        echo "TensorFlow git hash: $(git rev-parse HEAD)"

        echo "Building jaxlib..."
        cd ~/jax
        python3 build/build.py --enable_tpu --bazel_options="--override_repository=org_tensorflow=$HOME/tensorflow"
        pip install dist/*.whl
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'

        echo "Installing latest libtpu-nightly..."
        pip install libtpu-nightly \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
    },
  },

  jaxlibLatest:: {
    jaxlibVersion:: 'latest',
    scriptConfig+: {
      installLocalJax: |||
        cd ~/jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -r build/test-requirements.txt

        pip install .[tpu] \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      installLatestJax: |||
        pip install "jax[tpu]>=0.2.16" \
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
        # b/192016388: fix host_callback_to_tf_test.py
        pip install tensorflow
      ||| + self.maybeInstallLibtpuV4,
      maybeInstallLibtpuV4: if config.accelerator.type == 'tpu' && config.accelerator.version == 4 then |||
        gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev20211028-py3-none-any.whl .
        pip install libtpu_tpuv4-0.1.dev20211028-py3-none-any.whl
      ||| else '',
    },
  },
}
