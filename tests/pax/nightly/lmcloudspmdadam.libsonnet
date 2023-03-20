local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmspmdadam = common.NightlyPaxTest + common.Functional {
    modelName:: 'lmcloudspmdadam',
    expPath:: 'tasks.lm.params.c4.LmCloudSpmdAdamLimitSteps',
  },
  local v4_16 = {
    accelerator: tpus.v4_16,
  },
  configs: [
    lmspmdadam + v4_16,
  ],
}
