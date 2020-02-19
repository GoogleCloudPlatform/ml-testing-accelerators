local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local transformer = base.PyTorchTest {
    modelName: "fairseq-transformer",
    command: [
      "python3",
      "/tpu-examples/deps/fairseq/train.py",
      "/datasets/wmt18_en_de_bpej32k",
      "--tensorboard-logdir=$(MODEL_DIR)",
      "--metrics_debug",
      "--arch=transformer_vaswani_wmt_en_de_big",
      "--max-target-positions=64",
      "--attention-dropout=0.1",
      "--no-progress-bar",
      "--no-save",
      "--save-interval=1",
      "--criterion=label_smoothed_cross_entropy",
      "--source-lang=en",
      "--lr-scheduler=inverse_sqrt",
      "--min-lr=1e-09",
      "--skip-invalid-size-inputs-valid-test",
      "--target-lang=de",
      "--label-smoothing=0.1",
      "--update-freq=1",
      "--optimizer=adam",
      "--adam-betas=(0.9,0.98)",
      "--warmup-init-lr=1e-07",
      "--lr=0.0005",
      "--warmup-updates=4000",
      "--share-all-embeddings",
      "--dropout=0.3",
      "--weight-decay=0.0",
      "--valid-subset=valid",
      "--num_cores=8",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          volumes+: [
            {
              name: "wmt18-pd",
              gcePersistentDisk: {
                pdName: "wmt18-en-de-pd-central1-b",
                fsType: "ext4",
                readOnly: true,
              },
            },
          ],
          containers: [
            container {
              volumeMounts+: [{
                mountPath: "/datasets",
                name: "wmt18-pd",
                readOnly: true,
              }],
              resources+: {
                requests: {
                  cpu: "9.0",
                  memory: "30Gi",
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
      "--max-epoch=1",
      "--log_steps=10",
      "--train-subset=valid",
      "--input_shapes",
      "128x64",
    ],
    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              envMap+: {
                XLA_USE_BF16: "1",
              },
	    },
          },
        },
      },
    },
  },
  local convergence = mixins.Convergence {
    command+: [
      "--max-epoch=3",
      "--log_steps=100",
      "--train-subset=train",
      "--input_shapes",
      "64x64",
      "128x32",
      "256x16",
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        "loss_final": {
          success_threshold: {
            fixed_value: 4.5,
          },
          comparison: "less",
        },
      },
    },
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
  },
  configs: [
    transformer + v3_8 + functional + timeouts.Hours(1),
    transformer + v3_8 + convergence + timeouts.Hours(5),
  ],
}
