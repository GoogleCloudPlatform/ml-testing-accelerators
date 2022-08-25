local tpus = import 'templates/tpus.libsonnet';
local common = import '../common.libsonnet';
local utils = import 'templates/utils.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  local lmspmd2b = common.PaxTest +  mixins.Functional {
    modelName: 'lmspmd2b',
    // mode: 'example',
    // timeout: 3600, # 1 hour
    outputBucket: 'gs://pax-on-cloud-tpu-project',
    // Run every hour, on the hour
    // schedule: '0 */1 * * *',

    command: utils.scriptCommand(
      |||
        export GCS_BUCKET=$(MODEL_DIR)
        python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=tasks.lm.params.lm_cloud.LmCloudSpmd2B --job_log_dir=$(GCS_BUCKET)
      |||
    ),
  },
  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmspmd2b + v4_8,
  ],
}

