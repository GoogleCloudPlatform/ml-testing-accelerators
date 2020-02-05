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
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "datasets-pd",
                readOnly: true,
              }],
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
  },
  local convergence = mixins.Convergence {
    accelerator+: tpus.Preemptible,
    command+: [
      "--num_epochs=90",
      "--datadir=/datasets/imagenet",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    resnet50 + v3_8 + convergence + timeouts.Hours(23),
    resnet50 + v3_8 + functional + timeouts.Hours(2),
  ],
}
