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

local common = import '../../common.libsonnet';
local experimental_multipod = import '../experimental_multipod.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  PaxTest:: common.CloudAcceleratorTest + experimental_multipod.BaseTpuVmMixin {
    local config = self,

    frameworkPrefix: 'mp-pax',
    image: 'google/cloud-sdk',
    accelerator: tpus.v4_16,

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

    paxVersion:: error 'Add paxlib version mixin',
    scriptConfig:: {
      maybeBuildJaxlib: error 'Must define `maybeBuildJaxlib`',
      installLocalJax: error 'Must define `installLocalJax`',
      installLatestJax: error 'Must define `installLatestJax`',
      installPax: error 'Must define `installPax`',
      testEnvWorkarounds: error 'Must define `testEnvWorkarounds`',
      installPipPackages: |||
        echo "Installing Numpy, six and wheel"
        pip install numpy six wheel
      |||,
      printDiagnostics: |||
        python3 -c 'import jax; print("jax version:", jax.__version__)'
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
        echo "Environment Variable List:"
        printenv
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
        set +x
        set -u

        cat > testsetup.sh << SCRIPT_EOF
        set +x
        set -u
        set -e

        # .bash_logout sometimes causes a spurious bad exit code, remove it.
        rm .bash_logout

        %(installPipPackages)s
        %(installJax)s
        %(installJaxlib)s
        %(installLibtpu)s
        %(installPax)s
        SCRIPT_EOF

        setup_process_ids=()

        SLICE_COUNT=$(cat /scripts/slice_count)
        ZONE=$(cat /scripts/zone)

        for (( i=0; i < ${SLICE_COUNT}; i++ )); do
          gcloud alpha compute tpus tpu-vm ssh cloud-tpu-multipod-dev@$(cat /scripts/tpu_name_${i}) \
          --zone=${ZONE} \
          --ssh-key-file=/scripts/id_rsa \
          --strict-host-key-checking=no \
          --internal-ip \
          --worker=all \
          --command "$(cat testsetup.sh)" >> output_testsetup_${i}.txt 2>&1 &

          setup_process_ids+=($!)
        done

        echo "LOGGER: Waiting for test setup to be installed on all TPU VM hosts in ${SLICE_COUNT} slices."

        for i in "${!setup_process_ids[@]}"; do
          wait ${setup_process_ids[$i]}
          if [[ $? -ne 0 ]]; then
            echo "LOGGER: Set up failed on slice_${i}. Here is the output:"
            cat output_testsetup_${i}.txt
            bash /scripts/cleanup.sh
            exit 1
          fi
        done

        echo "LOGGER: Test set up completed successfully on ${SLICE_COUNT} slices."

        test_script_process_ids=()

        cat > test_script.sh << TEST_SCRIPT_EOF
        %(testScript)s
        TEST_SCRIPT_EOF

        for (( i=0; i < ${SLICE_COUNT}; i++ )); do
          for (( j=0; j < $(cat /scripts/worker_count_slice_${i}); j++ )); do
            gcloud alpha compute tpus tpu-vm ssh cloud-tpu-multipod-dev@$(cat /scripts/tpu_name_${i}) \
            --zone=${ZONE} \
            --ssh-key-file=/scripts/id_rsa \
            --strict-host-key-checking=no \
            --internal-ip \
            --worker=${j} \
            --command "$(cat test_script.sh)" >> output_slice_${i}_worker_${j}.txt 2>&1 &

            test_script_process_ids+=($!)
          done
        done

        echo "LOGGER: Waiting for test scripts to be completed on all TPU VM hosts in ${SLICE_COUNT} slices."

        for i in "${!test_script_process_ids[@]}"; do
          wait ${test_script_process_ids[$i]}
          if [[ $? -ne 0 ]]; then
            SLICE=$((${i}/${SLICE_COUNT}))
            WORKER=$(( ${i} - (${SLICE} * ${SLICE_COUNT}) ))
            echo "LOGGER: Test script failed on slice_${SLICE} & worker_${WORKER}. Here is the output:"
            cat output_slice_${SLICE}_worker_${WORKER}.txt
            bash /scripts/cleanup.sh
            exit 1
          fi
        done

        echo "LOGGER: Test script completed successfully on all the TPU VM hosts of ${SLICE_COUNT} slices. Here is the output from Slice 0:"
        cat output_slice_0_worker_0.txt

        echo "LOGGER: Cleaning up the TPU VM resources:"

        sleep 60

        bash /scripts/cleanup.sh

        exit_code=$?

        exit $exit_code
      ||| % {testScript: config.testScript, installPipPackages: config.scriptConfig.installPipPackages, installJax: config.scriptConfig.installJax, installJaxlib: config.scriptConfig.installJaxlib, installLibtpu: config.scriptConfig.installLibtpu, installPax: config.scriptConfig.installPax},
    ],
  },

  paxStable:: {
    paxVersion:: 'stable',
    scriptConfig+: {
      // Install jax without jaxlib or libtpu deps
      installJax: |||
        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git

        cd jax
        pip install -r build/test-requirements.txt

        pip install -e .

        cd
      |||,
      installJaxlib: |||
        echo "Cloning Tensorflow..."
        git clone https://github.com/tensorflow/tensorflow

        echo "Installing jaxlib from HEAD..."
        cd ~/jax
        python3 build/build.py --enable_tpu --bazel_options="--override_repository=org_tensorflow=/home/cloud-tpu-multipod-dev/tensorflow"
        pip install dist/jaxlib-*-cp*-manylinux2014_x86_64.whl --force-reinstall --no-deps

        echo "Jaxlib installation completed..."
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
      |||,
      installLibtpu: |||
        /usr/bin/docker-credential-gcr configure-docker
        sudo bash /var/scripts/docker-login.sh

        sudo docker create --name libtpu_next gcr.io/cloud-tpu-v2-images-dev/libtpu_unsanitized:nightly "/bin/bash"
        sudo docker cp libtpu_next:_libtpu_next.so /lib/libtpu.so

        sudo docker rm libtpu_next
        echo "export TPU_LIBRARY_PATH=/lib/libtpu.so" >> ~/.profile
      |||,
      installPax: |||
	echo "Installing Stable Pax from PyPI"

	pip install praxis
	pip install paxml
      |||,
    },
  },

  paxNightly:: {
    paxVersion:: 'nightly',
    scriptConfig+: {
      // Install jax without jaxlib or libtpu deps
      installJax: |||
        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git

        cd jax
        pip install -r build/test-requirements.txt

        pip install -e .

        cd
      |||,
      installJaxlib: |||
        echo "Installing jaxlib from Nightly..."
        pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

        echo "Jaxlib installation completed..."
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
      |||,
      installLibtpu: |||
        /usr/bin/docker-credential-gcr configure-docker
        sudo bash /var/scripts/docker-login.sh

        sudo docker create --name libtpu_next gcr.io/cloud-tpu-v2-images-dev/libtpu_unsanitized:nightly "/bin/bash"
        sudo docker cp libtpu_next:_libtpu_next.so /lib/libtpu.so

        sudo docker rm libtpu_next
        echo "export TPU_LIBRARY_PATH=/lib/libtpu.so" >> ~/.profile
      |||,
      installPax: |||
        today=$(date + +%Y%m%d)
        echo "Installing nightly Pax"
        gsutil cp gs://pax-on-cloud-tpu-project/wheels/${today}/paxml*.whl .
        gsutil cp gs://pax-on-cloud-tpu-project/wheels/${today}/praxis*.whl .
        if [ -f praxis*.whl -a -f paxml*.whl ]; then
          echo "Nightly builds succeeded."
        else
          echo "Nighlty builds failed or are pending."
          exit 1
        fi
        
        pip install praxis*.whl
        pip install paxml*.whl
      |||,
    },
  },

  tpuVmV4Base:: {
    local config = self,
    accelerator: tpus.v4_16,

    tpuSettings+: {
      softwareVersion: 'tpu-vm-base-gvnic',
    },
    scriptConfig+: {
      testEnvWorkarounds: |||
        pip install tensorflow
        pip uninstall -y libtpu-nightly

      |||,
    },
  },
}
