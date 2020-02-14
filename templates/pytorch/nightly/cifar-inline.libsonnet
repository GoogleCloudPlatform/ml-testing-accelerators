local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local cifar = base.PyTorchTest {
    modelName: "cifar-inline",
    command: [
      "python3",
      "pytorch/xla/test/test_train_cifar.py",
      "--metrics_debug",
      "--target_accuracy=72",
      "--datadir=/datasets/cifar-data",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "cifar-pd",
              gcePersistentDisk: {
                pdName: "cifar-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "cifar-pd",
                readOnly: true,
              }],
            } for container in super.containers
          ],
        },
      },
    },
  },
  local convergence = mixins.Convergence {
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 72.0,
          },
          comparison: "greater",
        },
      },
    },
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    cifar + v2_8 + convergence + timeouts.Hours(1),
    cifar + v3_8 + convergence + timeouts.Hours(1),
  ],
}
