local common = import '../common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  Functional:: mixins.Functional {
    // Run at 3AM PST daily
    schedule: '0 11 * * *',
  },
  StablePaxTest:: common.PaxTest {
    local config = self,
    frameworkPrefix: 'pax-stable',
    expPath:: '',
    extraFlags:: [],

    // PAX tests are structured as bash scripts that run directly on the Cloud
    // TPU VM instead of using docker images
    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      # install Pax and dependencies
      pip install paxml

      pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      echo "num_devices: $num_devices"
      if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
      fi


      python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=%(expPath)s --job_log_dir=$(MODEL_DIR) %(extraFlags)s
    ||| % { expPath: config.expPath, extraFlags: std.join(' ', config.extraFlags) },

  },
  Convergence:: mixins.Convergence {
    // Run at 4AM PST daily
    schedule: '0 12 * * *',
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Metrics/log_pplx': {
              FINAL: {
                inclusive_bounds: true,
                fixed_value: {
                  comparison: 'LESS',
                  std_devs: 3.0,
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
