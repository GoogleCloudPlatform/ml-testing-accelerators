local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local c4spmd1b_pretraining = common.NightlyPaxTest + common.Convergence {
    modelName:: 'c4spmd1b-pretraining',
    expPath:: 'tasks.lm.params.c4.C4Spmd1BAdam4ReplicasLimitSteps',
    extraFlags:: [],
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    c4spmd1b_pretraining + v4_8,
  ],
}
