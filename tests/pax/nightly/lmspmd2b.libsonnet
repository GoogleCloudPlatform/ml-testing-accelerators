local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local lmspmd2b = common.NightlyPaxTest + common.Functional {
    modelName:: 'lmspmd2b',
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps',
    extraFlags:: ['--jax_fully_async_checkpoint=False'],
  },
  local lmspmd2b_ckpt = common.NightlyPaxTest + common.Functional {
    local config = self,
    expPath:: 'tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps',
    modelName:: 'lmspmd2b-ckpt',

    // PAX tests are structured as bash scripts that run directly on the Cloud
    // TPU VM instead of using docker images
    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      gsutil -m cp -r "$(PAX_DIR)/lmcloudspmd2B/pax-nightly-lmspmd2b-func-v4-8-1vm-run1/*" $(MODEL_DIR)

      # check for nightly build
      gsutil cp gs://pax-on-cloud-tpu-project/wheels/%(buildDate)s/paxml*.whl .
      gsutil cp gs://pax-on-cloud-tpu-project/wheels/%(buildDate)s/praxis*.whl .

      if [ -f praxis*.whl -a -f paxml*.whl ]; then
          echo "Nightly builds succeeded."
      else
          echo "Nighlty builds failed or are pending."
          exit 1
      fi

      # install Pax and dependencies
      pip install praxis*.whl
      pip install paxml*.whl
      sudo pip uninstall --yes jax jaxlib libtpu-nightly

      pip install git+https://github.com/google/jax.git
      pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
      pip install --no-index -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      echo "num_devices: $num_devices"
      if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
      fi

      python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=%(expPath)s --job_log_dir=$(MODEL_DIR) %(extraFlags)s
    ||| % { buildDate: config.buildDate, expPath: config.expPath, extraFlags: std.join(' ', config.extraFlags) },
  },

  local v4_8 = {
    accelerator: tpus.v4_8,
  },
  configs: [
    lmspmd2b + v4_8,
    lmspmd2b_ckpt + v4_8,
  ],
}
