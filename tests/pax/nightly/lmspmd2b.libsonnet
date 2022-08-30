local common = import '../common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local lmspmd2b = common.PaxTest +  mixins.Functional {
    modelName: 'lmspmd2b',

    command: utils.scriptCommand(
      |||
        %(common_install)s
        python3 /usr/local/lib/python3.7/dist-packages/paxml/main.py --exp=tasks.lm.params.lm_cloud.LmCloudSpmd2B --job_log_dir=$(MODEL_DIR)
        %(common_cleanup)s
      ||| % { common_install: common.pax_install, common_cleanup: common.cleanup }
    ),
  },
  configs: [
    lmspmd2b,
  ],
}
