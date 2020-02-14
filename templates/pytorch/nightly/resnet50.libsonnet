local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local resnet50 = base.PyTorchTest {
    modelName: "resnet50",
    command: [
      "python3",
      "pytorch/xla/test/test_train_imagenet.py",
      "--model=resnet50",
      "--num_workers=64",
      "--batch_size=128",
      "--log_steps=200",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          containers: [
            container {
              resources+: {
                requests: {
                  cpu: "90.0",
                  memory: "400Gi",
                },
              },
            } for container in super.containers
          ],
        },
      },
    },
  },
  local functional = mixins.Functional {
    command+: [
      "--num_epochs=2",
      "--datadir=/datasets/imagenet-mini",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "imagenet-mini-pd",
              gcePersistentDisk: {
                pdName: "imagenet-mini-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "imagenet-mini-pd",
                readOnly: true,
              }],
            } for container in super.containers
          ],
        },
      },
    },
  },
  local convergence = mixins.Convergence {
    accelerator+: tpus.Preemptible,
    command+: [
      "--num_epochs=90",
      "--datadir=/datasets/imagenet",
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 76.0,
          },
          comparison: "greater",
        },
      },
    },
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "imagenet-pd",
              gcePersistentDisk: {
                pdName: "imagenet-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "imagenet-pd",
                readOnly: true,
              }],
            } for container in super.containers
          ],
        },
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    resnet50 + v3_8 + convergence + timeouts.Hours(23),
    resnet50 + v3_8 + functional + timeouts.Hours(2),
  ],
}
