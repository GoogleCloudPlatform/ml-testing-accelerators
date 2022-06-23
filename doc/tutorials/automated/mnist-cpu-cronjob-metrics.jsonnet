local base = import 'templates/base.libsonnet';
local metrics = import 'templates/metrics.libsonnet';

local mnist = base.BaseTest {
  // Configure job name
  frameworkPrefix: "tf",
  modelName: "mnist",
  mode: "example",
  timeout: 3600, # 1 hour, in seconds

  // Set up runtime environment
  image: 'tensorflow/tensorflow', // Official TF docker image
  imageTag: '2.9.1',
  accelerator: base.cpu,
  outputBucket: std.extVar('gcs-bucket'),

  // Run every hour, on the hour
  schedule: '0 */1 * * *',

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
    '--data_dir=/tmp/mnist',
    '--download',
    '--num_gpus=0',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],

  metricConfig: metrics.MetricCollectionConfigHelper {
    sourceMap:: {
      tensorboard: metrics.TensorBoardSourceHelper {
        include_tags: [
          {
            tag_pattern: '*',
            strategies: [
              'FINAL',
            ],
          },
        ],
      },
      literals: {
        assertions: {
          duration: {
            std_devs_from_mean: {
              comparison: 'LESS',
              std_devs: 2,
            },
          },
        },
      },
    },
  },
};

std.manifestYamlDoc(mnist.cronJob, quote_keys=false)
