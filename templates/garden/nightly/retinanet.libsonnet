local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";

{
  local retinanet = base.GardenTest {
    modelName: "retinanet",
    params:: {
      trainSteps: error "Must set `trainSteps`",
      trainBatchSize: error "Must set `trainBatchSize`",
    },
    command: [
      "python3",
      "official/vision/detection/main.py",
      "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)",
      "--strategy_type=tpu",
      '--params_override="%s"' % |||
        eval:
          eval_file_pattern: 'gs://xl-ml-test-us-central1/data/coco/val*'
          batch_size: 8
          val_json_file: 'gs://xl-ml-test-us-central1/data/coco/instances_val2017.json'
        predict:
          predict_batch_size: 8
        architecture:
          use_bfloat16: True
        retinanet_parser:
          use_bfloat16: True
        train:
          checkpoint:
            path: 'gs://xl-ml-test-us-central1/data/pretrain/resnet50-checkpoint-2018-02-07'
            prefix: 'resnet50/'
          total_steps: %(trainSteps)s
          batch_size: %(trainBatchSize)s
          train_file_pattern: 'gs://xl-ml-test-us-central1/data/coco/train*'";
      ||| % self.params,
    ],
  },
  local functional = mixins.Functional {
    command+: [
      "--mode=train",
    ],
    params+: {
      trainSteps: 1000,
    },
  },
  local convergence = mixins.Convergence {
    command+: [
      "--mode=train_and_eval",
    ],
    params+: {
      trainSteps: 22500,
    },
  },
  local v2_8 = {
    accelerator+: tpus.v2_8,
    params+: {
      trainBatchSize: 64,
    },
  },
  local v3_8 = {
    accelerator+: tpus.v3_8,
    params+: {
      trainBatchSize: 64,
    },
  },

  configs: [
    retinanet + functional + v2_8,
    retinanet + functional + v3_8,
    retinanet + convergence + v2_8,
    retinanet + convergence + v3_8,
  ],
}