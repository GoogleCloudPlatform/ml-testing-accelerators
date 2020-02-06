local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local tpus = import "../../tpus.libsonnet";
local gpus = import "../../gpus.libsonnet";

{
  local mnist = base.GardenTest {
    modelName: "mnist",
    command: [
      "python3",
      "official/vision/image_classification/mnist_main.py",
      "--data_dir=gs://xl-ml-test-us-central1/data/mnist",
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--train_epochs=1",
      "--epochs_between_evals=1",
    ],
  },
  local convergence = mixins.Convergence {
    command+: [
      "--train_epochs=10",
      "--epochs_between_evals=10",
    ],
  },
  local v100 = {
    accelerator: gpus.teslaV100,
    command+: [
      "--num_gpus=1",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--distribution_strategy=tpu",
      "--batch_size=1024",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--distribution_strategy=tpu",
      "--batch_size=2048",
    ],
  },

  configs: [
    mnist + v100 + functional,
    mnist + v2_8 + functional,
    mnist + v2_8 + convergence,
    mnist + v3_8 + functional,
    mnist + v3_8 + convergence,
  ],
}