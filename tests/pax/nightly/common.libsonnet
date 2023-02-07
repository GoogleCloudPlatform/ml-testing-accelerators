local common = import '../common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  Functional:: mixins.Functional {
    // Run at 2AM PST daily
    schedule: '0 10 * * *',
  },
  NightlyPaxTest:: common.PaxTest {
    local config = self,

    tpuSettings+: {
          softwareVersion: 'nightly',
        },

    expPath:: '',
    extraFlags:: [],
    buildDate:: '$(date +%Y%m%d)',

    // PAX tests are structured as bash scripts that run directly on the Cloud
    // TPU VM instead of using docker images
    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

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

      # need to install chex from source, since pip version is currently incompatible with latest JAX
      pip install -U git+https://github.com/deepmind/chex.git

      pip install git+https://github.com/google/jax.git
      pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
      pip install -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      pip install protobuf==3.15

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      echo "num_devices: $num_devices"
      if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
      fi

      python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=%(expPath)s --job_log_dir=logs %(extraFlags)s
    ||| % { buildDate: config.buildDate, expPath: config.expPath, extraFlags: std.join(' ', config.extraFlags) },
  },
  Convergence:: mixins.Convergence {
    // Run at 2AM PST daily
    schedule: '0 10 * * *',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Metrics/log_pplx': {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'LESSER',
                  std_devs: 2.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
}
