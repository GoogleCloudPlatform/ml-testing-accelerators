local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local transformer = base.GardenTest {
    modelName: "transformer-translate",
    command: [
      "python3",
      "official/transformer/v2/transformer_main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--steps_between_evals=10000",
      "--static_batch=true",
      "--use_ctl=true",
      "--param_set=big",
      "--max_length=64",
      "--decode_batch_size=32",
      "--decode_max_length=97",
      "--padded_decode=true",
      "--distribution_strategy=tpu",
      "--data_dir=gs://xl-ml-test-us-central1/data/transformer",
      "--vocab_file=gs://xl-ml-test-us-central1/data/transformer/vocab.ende.32768",
      "--bleu_source=gs://xl-ml-test-us-central1/data/transformer/newstest2014.en",
      "--bleu_ref=gs://xl-ml-test-us-central1/data/transformer/newstest2014.de",
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--train_steps=20000",
    ],
  },
  local convergence = mixins.Convergence {
    command+: [
      "--train_steps=200000",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--batch_size=6144",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--batch_size=6144",
    ],
  },

  configs: [
    transformer + functional + v2_8,
    transformer + functional + v3_8,
    transformer + convergence + v2_8,
    transformer + convergence + v3_8,
  ],
}