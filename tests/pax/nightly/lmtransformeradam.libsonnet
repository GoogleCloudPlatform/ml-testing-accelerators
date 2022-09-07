local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local lmtransformeradam = common.NightlyPaxTest + mixins.Functional {
    modelName: 'lmtransformeradam',
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudTransformerAdamLimitSteps',
    extraFlags:: '--jax_fully_async_checkpoint=False --pmap_use_tensorstore=True',
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmtransformeradam + v4_8,
  ],
}
