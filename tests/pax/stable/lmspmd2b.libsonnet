local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmspmd2b = common.StablePaxTest + common.Functional {
    modelName:: 'lmspmd2b',
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps',
    extraFlags:: [],
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmspmd2b + v4_8,
  ],
}
