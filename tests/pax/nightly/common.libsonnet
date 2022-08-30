local common = import '../common.libsonnet';

{
    NightlyPaxTest:: common.PaxTest {
        local config = self,
        expPath:: '',
        extraFlags:: '',

        // PAX tests are structured as bash scripts that run directly on the Cloud
        // TPU VM instead of using docker images
        testScript:: |||
            set -x
            set -u
            set -e

            # .bash_logout sometimes causes a spurious bad exit code, remove it.
            rm .bash_logout

            gsutil cp gs://pax-on-cloud-tpu-project/wheels/20220830/paxml*.whl .
            gsutil cp gs://pax-on-cloud-tpu-project/wheels/20220830/praxis*.whl .
            pip install praxis*.whl
            pip install paxml*.whl
            pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
            pip install protobuf==3.15

            num_devices=`python3 -c "import jax; print(jax.device_count())"`
            echo "num_devices: $num_devices"
            if [ "$num_devices" = "1" ]; then
                echo "No TPU devices detected"
                exit 1
            fi

            python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=%(expPath)s --job_log_dir=$(MODEL_DIR) %(extraFlags)s
        ||| % config,
    },
}