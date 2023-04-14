local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmtransformeradam = self.lmtransformeradam,
  lmtransformeradam:: common.NightlyPaxTest + common.Functional {
    modelName: 'lmtransformeradam',
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudTransformerAdamLimitSteps',
    extraFlags:: ['--jax_fully_async_checkpoint=False', '--pmap_use_tensorstore=True'],
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },
  local v4_16 = self.v4_16,
  v4_16:: {
    accelerator: tpus.v4_16,
  },
  configs: [
    lmtransformeradam + v4_8,
    lmtransformeradam + v4_16,
  ],
}
