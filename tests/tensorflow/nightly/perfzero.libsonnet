local common = import "common.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local gpus = import "templates/gpus.libsonnet";
local utils = import "templates/utils.libsonnet";

{
  local PerfZeroTest = common.ModelGardenTest {
    local config = self,

    timeout: timeouts.one_hour,
    schedule: "0 0 * * *",
    mode: "perfzero",
    imageTag: "perfzero",

    benchmarkOptions:: {
      test: {
        module: null,
        class: null,
        method: null,
      },
      table: {
        project: "xl-ml-test",
        name: "perfzero_dataset.perfzero_table",
      },
    },
    command: [
      "python3",
      "/benchmarks/perfzero/lib/benchmark.py",
      "--gcloud_key_file=",
      "--bigquery_project_name=%s" % config.benchmarkOptions.table.project,
      "--bigquery_dataset_table_name=%s" % config.benchmarkOptions.table.name,
      "--benchmark_methods=%(module)s.%(class)s.%(method)s" % config.benchmarkOptions.test,
      "--output_gcs_url=$(MODEL_DIR)",
    ],

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+:: {
            train+: {
              envMap+:: {
                BENCHMARK_OUTPUT_DIR: "$(MODEL_DIR)",
              },
              args: utils.scriptCommand(
                # HACK: Replace some hard-coded data paths
                |||
                  sed -i 's_gs://tf-perfzero-data/bert_$(PERFZERO_DATA_DIR)_g' /garden/official/benchmark/bert_squad_benchmark.py
                  sed -i 's_gs://tf-perfzero-data_$(PERFZERO_DATA_DIR)_g' /garden/official/benchmark/retinanet_benchmark.py
                  sed -i 's_gs://mlcompass-data/transformer_$(PERFZERO_DATA_DIR)_g' /garden/official/benchmark/transformer_benchmark.py
                  sed -i 's_gs://mlcompass-data/imagenet/imagenet-2012-tfrecord_$(PERFZERO_DATA_DIR)/imagenet_g' /garden/official/benchmark/resnet_ctl_imagenet_benchmark.py
                  sed -i 's/wmt32k-en2de-official/transformer/g' /garden/official/benchmark/transformer_benchmark.py

                  if [ -v TPU_NAME ]; then
                    export BENCHMARK_TPU=${TPU_NAME#*/}
                  fi

                  %s
                ||| % std.join(" ", config.command))
            },
          },
        },
      },
    },
  },

  # Common benchmark methods.
  local benchmark_1_gpu = {
    accelerator: gpus.teslaV100,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_1_gpu",
      },
    },
  },
  local benchmark_1_gpu_no_dist_strat = {
    accelerator: gpus.teslaV100,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_1_gpu_no_dist_strat",
      },
    },
  },
  local benchmark_8_gpu = {
    accelerator: gpus.teslaV100 + { count: 8 },
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_8_gpu",
      },
    },
  },
  local benchmark_2x2_tpu = {
    accelerator: tpus.v3_8,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_2x2_tpu",
      },
    },
  },
  local benchmark_2x2_tpu_bf16 = {
    accelerator: tpus.v3_8,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_2x2_tpu_bf16",
      },
    },
  },
  local benchmark_4x4_tpu = {
    accelerator: tpus.v3_32,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_4x4_tpu",
      },
    },
  },
  local benchmark_4x4_tpu_bf16 = {
    accelerator: tpus.v3_32,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_4x4_tpu_bf16",
      },
    },
  },

  local bertSquad = PerfZeroTest {
    modelName: "bert-squad",
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.bert_squad_benchmark",
        class: "BertSquadBenchmarkReal",
      },
    },
  },

  local resnet50Keras = PerfZeroTest {
    modelName: "resnet50-cfit",
    command+: [
      # TODO: replace with env
      "--root_data_dir=$(PERFZERO_DATA_DIR)"
    ],
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.keras_imagenet_benchmark",
        class: "Resnet50KerasBenchmarkReal",
      },
    },
  },
  
  local resnet50Ctl = PerfZeroTest {
    modelName: "resnet50-ctl",
    command+: [
      # TODO: replace with env
      "--root_data_dir=$(PERFZERO_DATA_DIR)"
    ],
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.resnet_ctl_imagenet_benchmark",
        class: "Resnet50CtlBenchmarkReal",
      },
    },
  },

  local efficientnetKeras = PerfZeroTest {
    modelName: "efficientnet",
    command+: [
      "--root_data_dir=$(PERFZERO_DATA_DIR)"
    ],
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.keras_imagenet_benchmark",
        class: "EfficientNetKerasBenchmarkReal",
      },
    },
  },

  # Detection benchmark names have a *_coco suffix.
  local coco = {
    benchmarkOptions+:: {
      test+: {
        method+: "_coco",
      },
    },
  },
  local retinanet = PerfZeroTest {
    modelName: "retinanet",
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.retinanet_benchmark",
        class: "RetinanetBenchmarkReal",
      },
    },
  },
  local maskrcnn = PerfZeroTest {
    modelName: "maskrcnn",
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.retinanet_benchmark",
        class: "RetinanetBenchmarkReal",
      },
    },
  },
  local shapemask = PerfZeroTest {
    modelName: "shapemask",
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.retinanet_benchmark",
        class: "RetinanetBenchmarkReal",
      },
    },
  },

  # Transformer multi-GPU methods have '_static_batch' suffix.
  local static_batch = {
    benchmarkOptions+:: {
      method+: "_static_batch",
    },
  },
  local transformer = PerfZeroTest {
    modelName: "transformer",
    command+: [
      "--root_data_dir=$(PERFZERO_DATA_DIR)"
    ],
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.transformer_benchmark",
        class: "TransformerKerasBenchmark",
      },
    },
  },
  local transformerBig = PerfZeroTest {
    modelName: "transformer",
    command+: [
      "--root_data_dir=$(PERFZERO_DATA_DIR)"
    ],
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.transformer_benchmark",
        class: "TransformerBigKerasBenchmarkReal",
      },
    },
  },

  configs: [
    bertSquad + benchmark_1_gpu,
    bertSquad + benchmark_8_gpu,
    bertSquad + benchmark_2x2_tpu,
    resnet50Ctl + benchmark_1_gpu_no_dist_strat,
    resnet50Ctl + benchmark_8_gpu,
    resnet50Ctl + benchmark_2x2_tpu_bf16,
    resnet50Ctl + benchmark_4x4_tpu_bf16,
    resnet50Keras + benchmark_1_gpu_no_dist_strat,
    resnet50Keras + benchmark_8_gpu,
    resnet50Keras + benchmark_2x2_tpu_bf16,
    resnet50Keras + benchmark_4x4_tpu_bf16,
    efficientnetKeras + benchmark_1_gpu_no_dist_strat,
    efficientnetKeras + benchmark_8_gpu,
    efficientnetKeras + benchmark_2x2_tpu_bf16,
    efficientnetKeras + benchmark_4x4_tpu_bf16,
    retinanet + benchmark_1_gpu + coco,
    retinanet + benchmark_8_gpu + coco + timeouts.Hours(2),
    retinanet + benchmark_2x2_tpu + coco,
    retinanet + benchmark_4x4_tpu + coco,
    maskrcnn + benchmark_1_gpu + coco,
    maskrcnn + benchmark_8_gpu + coco,
    maskrcnn + benchmark_2x2_tpu + coco,
    maskrcnn + benchmark_4x4_tpu + coco,
    shapemask + benchmark_1_gpu + coco,
    shapemask + benchmark_8_gpu + coco,
    shapemask + benchmark_2x2_tpu + coco,
    shapemask + benchmark_4x4_tpu + coco,
    transformer + benchmark_1_gpu,
    transformer + benchmark_8_gpu + static_batch + timeouts.Hours(2.5),
    transformerBig + benchmark_2x2_tpu,
    transformerBig + benchmark_4x4_tpu,
  ],
}