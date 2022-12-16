local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmSpmdTest = common.PaxTest + mixins.Functional {
    local config = self,
    modelName: '%s-lmspmd' % [self.paxVersion],
    frameworkPrefix: 'mk',
    tpuSettings+: {
      slices: 2,
    },
    
    // Trigger the test at 09:00 UTC.
    schedule: '0 9 * * *',
    gcsPath:: 'gs://xlml-logs/lmcloud_multislice',
    modelConf:: |||
      set +x
      set -u
      set -e
      
      git clone --single-branch --branch multipod-tests https://github.com/GoogleCloudPlatform/ml-testing-accelerators.git --depth=1
      cat ~/ml-testing-accelerators/tests/multipods/pax/model_confs/lm_cloud.py >> ~/.local/lib/python3.8/site-packages/paxml/tasks/lm/params/lm_cloud.py
      exit 0
    |||,
    testScript:: |||
      set +x
      set -u
      set -e
      . ~/.profile
      export JAX_USE_PJRT_C_API_ON_TPU=1
      export TPU_STDERR_LOG_LEVEL=0
      export TPU_MIN_LOG_LEVEL=0
      export TF_CPP_MIN_LOG_LEVEL=0
      export TPU_NAME=local
      
      python3 ~/.local/lib/python3.8/site-packages/paxml/main.py --exp=tasks.lm.params.lm_cloud.LmCloudSpmdMultislice2BXLML --job_log_dir=%(gcsPath)s/logs --enable_checkpoint_saving=false
    ||| % { gcsPath: config.gcsPath },
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmSpmdTest + common.paxNightly + common.tpuVmV4Base + v4_8,
  ],
}
