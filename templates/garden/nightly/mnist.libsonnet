local base = import 'base.libsonnet';
local modes = import '../../modes.libsonnet';
local tpus = import '../../tpus.libsonnet';

{
  local mnist = base.GardenTest {
    model_name: 'mnist',
    command: [
      'python3',
      'official/vision/image_classification/mnist_main.py',
      '--data_dir=gs://xl-ml-test-us-central1/data/mnist',
      '--distribution_strategy=tpu',
      '--clean',
    ],
  },
  local functional = modes.Functional {
    command+: [
      '--train_epochs=1',
      '--epochs_between_evals=1',
    ],
  },
  local convergence = modes.Convergence {
    command+: [
      '--train_epochs=10',
      '--epochs_between_evals=10',
    ],
  },
  local v2_8 = {
    accelerator+: tpus.v2_8,
    command+: [ '--batch_size=1024' ],
  },
  local v3_8 = {
    accelerator+: tpus.v3_8,
    command+: [ '--batch_size=2048' ],
  },

  configs: [
    mnist + v2_8 + functional,
    mnist + v2_8 + convergence,
    mnist + v3_8 + functional,
    mnist + v3_8 + convergence,
  ],
}