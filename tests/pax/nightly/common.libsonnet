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


      # check for nightly build
      export DATE=$(date +%Y%m%d)
      gsutil cp gs://pax-on-cloud-tpu-project/wheels/$DATE/paxml*.whl .
      gsutil cp gs://pax-on-cloud-tpu-project/wheels/$DATE/praxis*.whl .

      if [ -f praxis*.whl -a -f paxml*.whl ]; then
          echo "Nightly build succeeded."
      else 
          echo "Nighlty build failed or is pending."
          exit 1
      fi 

      # install pax and dependencies
      pip install -U pip
      pip install praxis*.whl
      pip install paxml*.whl
      sudo pip uninstall jax jaxlib libtpu-nightly libtpu libtpu_tpuv4 tensorflow  -y
      git clone https://github.com/google/jax.git
      cd jax/
      pip install .
      cd ..
      pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
      pip install -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      mkdir -p libtpu_folder
      gsutil -m cp -R gs://pax-on-cloud-tpu-project/libtpu/libtpu.so libtpu_folder/
      ls -l libtpu_folder/
      cp libtpu_folder/libtpu.so .local/lib/python3.8/site-packages/libtpu/libtpu.so
      pip install protobuf==3.15

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      echo "num_devices: $num_devices"
      if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
      fi

      export GCS_BUCKET=MODEL_DIR

      python3 .local/lib/python3.8/site-packages/paxml/main.py --exp=%(expPath)s --job_log_dir=logs %(extraFlags)s
    ||| % config,
  },
}
