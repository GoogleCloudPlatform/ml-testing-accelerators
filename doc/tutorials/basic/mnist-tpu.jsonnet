local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

local mnist = base.BaseTest {
  // Configure job name
  frameworkPrefix: "tf",
  modelName: "mnist",
  mode: "example",
  timeout: 3600, # 1 hour, in seconds

  // Set up runtime environment
  image: 'tensorflow/tensorflow', // Official TF docker image
  imageTag: '2.9.1',
  tpuSettings+: {
    softwareVersion: '2.9.1',
  },
  accelerator: tpus.v2_8,
  outputBucket: std.extVar('gcs-bucket'),

  // Override entrypoint to install TF official models before running `command`.
  entrypoint: [
    'bash',
    '-c',
    |||
      pip install tf-models-official==2.9.1

      # Run whatever is in `command` here
      ${@:0}
    |||
  ],
  command: [
    'python3',
    '-m',
    'official.legacy.image_classification.mnist_main',
    '--distribution_strategy=tpu',
    '--data_dir=$(MODEL_DIR)/data',
    '--download',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],
};

std.manifestYamlDoc(mnist.oneshotJob, quote_keys=false)
