local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local syncDevicesTest = common.PaxTest + mixins.Functional {
    modelName: '%s-sync-devices' % [self.paxmlVersion],
    accelerator: tpus.v4_8,
    tpuSettings+: {
      slices: 4,
    },
    // Trigger the test at 09:00 UTC.
    schedule: '0 9 * * *',

    testScript:: |||
        set +x
      	set -u
      	set -e
      	. ~/.profile
      	export TPU_NAME=local
      	export TPU_STDERR_LOG_LEVEL=0
      	export TPU_MIN_LOG_LEVEL=0
      	export JAX_USE_PJRT_C_API_ON_TPU=1
      	export TF_CPP_MIN_LOG_LEVEL=0

	git clone --single-branch --branch multipod-tests https://github.com/GoogleCloudPlatform/ml-testing-accelerators.git --depth=1
      	python3 ml-testing-accelerators/tests/multipods/pax/unit_tests/sync_devices.py
      	exit 0
      |||,
    },

    configs: [
	syncDevicesTest + common.jaxlibNightly + common.tpuVmV4Base,
    ],
}
