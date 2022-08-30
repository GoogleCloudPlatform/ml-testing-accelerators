local common = import '../common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local lmtransformeradam = common.PaxTest + mixins.Functional {
    modelName: 'lmtransformeradam',

    command: utils.scriptCommand(
      |||
        %(common_install)s
        python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=tasks.lm.params.lm_cloud.LmCloudSpmd2B --job_log_dir=$(MODEL_DIR)
        %(common_cleanup)s
      ||| % { common_install: common.pax_install, common_cleanup: common.cleanup }
    ),
  },
  configs: [
    lmtransformeradam,
  ],
}

