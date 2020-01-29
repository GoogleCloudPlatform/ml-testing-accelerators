local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local bert = base.GardenTest {
    modelName: "bert-mnli",
    command: [
      "python3",
      "official/nlp/bert/run_classifier.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--steps_per_loop=1000",
      "--input_meta_data_path=gs://cloud-tpu-checkpoints/bert/classification/mnli_meta_data",
      "--train_data_path=gs://cloud-tpu-checkpoints/bert/classification/mnli_train.tf_record",
      "--eval_data_path=gs://cloud-tpu-checkpoints/bert/classification/mnli_eval.tf_record",
      "--bert_config_file=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_config.json",
      "--init_checkpoint=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16/bert_model.ckpt",
      "--learning_rate=3e-5",
      "--distribution_strategy=tpu",
    ],
  },
  local functional = mixins.Functional {
    command+: [
      '--num_train_epochs=1',
    ],
  },
  local convergence = mixins.Convergence {
    command+: [
      '--num_train_epochs=3',
    ],
  },
  local v2_8 = {
    accelerator+: tpus.v2_8,
    command+: [
      '--train_batch_size=32',
      '--eval_batch_size=32',
    ],
  },
  local v3_8 = {
    accelerator+: tpus.v3_8,
    command+: [
      '--train_batch_size=32',
      '--eval_batch_size=32',
    ],
  },

  configs: [
    bert + v2_8 + functional,
    bert + v3_8 + functional,
    bert + v2_8 + convergence + timeouts.Hours(2),
    bert + v3_8 + convergence + timeouts.Hours(2),
  ],
}