local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmspmd2b = common.NightlyPaxTest +  mixins.Functional {
    modelName:: 'lmspmd2b',
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps',
    extraFlags:: '--jax_fully_async_checkpoint=False',
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmspmd2b + v4_8,
  ],
}
