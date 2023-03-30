local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmspmd2b = common.NightlyPaxTest + common.Functional {
    modelName:: 'lmspmd2b',
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps',
    extraFlags:: ['--jax_fully_async_checkpoint=False'],
  },
  local lmspmd2b_ckpt = lmspmd2b {
    modelName:: 'lmspmd2b-ckpt',
    testScript+: |||
    gsutil -m cp "$(PAX_DIR)/lmcloudspmd2B/pax-nightly-lmspmd2b-func-v4-8-1vm-run1/\DIR*" $(MODEL_DIR)
    |||
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmspmd2b + v4_8,
    lmspmd2b_ckpt + v4_8,
  ],
}
