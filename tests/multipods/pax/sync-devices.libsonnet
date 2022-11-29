local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local syncDevicesTest = common.PaxTest + mixins.Functional {
    modelName: '%s-sync-devices' % [self.paxVersion],
    tpuSettings+: {
      slices: 2,
      accelerator: tpus.v4_8,
    },
    // Trigger the test at 09:00 UTC.
    schedule: '0 9 * * *',

    testScript:: |||
      set +x
      set -u
      set -e
      . ~/.profile

      %(printDiagnostics)s

      export TPU_NAME=local
      export TPU_STDERR_LOG_LEVEL=0
      export JAX_USE_PJRT_C_API_ON_TPU=1

      git clone --single-branch --branch multipod-tests https://github.com/GoogleCloudPlatform/ml-testing-accelerators.git --depth=1
      python3 ml-testing-accelerators/tests/multipods/pax/unit_tests/sync_devices.py

    ||| % self.scriptConfig,
  },
  configs: [
    syncDevicesTest + common.paxNightly + common.tpuVmV4Base,
  ],
}
