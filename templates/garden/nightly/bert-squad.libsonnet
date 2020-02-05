local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local bert = base.GardenTest {
    modelName: "bert-squad",
    command: [
      "python3",
      "official/nlp/bert/run_squad.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--input_meta_data_path=gs://xl-ml-test-us-central1/data/squad/squad_v1.1_meta_data.json",
      "--train_data_path=gs://xl-ml-test-us-central1/data/squad/squad_v1.1_train.tfrecord",
      "--predict_file=gs://xl-ml-test-us-central1/data/squad/dev-v1.1.json",
      "--vocab_file=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/vocab.txt",
      "--bert_config_file=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_config.json",
      "--init_checkpoint=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_model.ckpt",
      "--learning_rate=8e-5",
      "--do_lower_case=true",
      "--distribution_strategy=tpu",
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--mode=train",
      "--num_train_epochs=1",
    ],
  },
  local convergence = mixins.Convergence {
    command+: [
      "--mode=train_and_predict",
      "--num_train_epochs=2",
    ],
  },
  local v2_8 = {
    accelerator: tpus.v2_8,
    command+: [
      "--train_batch_size=16",
    ],
  },
  local v3_8 = {
    accelerator: tpus.v3_8,
    command+: [
      "--train_batch_size=32",
    ],
  },

  configs: [
    bert + v2_8 + functional,
    bert + v3_8 + functional,
  ],
}