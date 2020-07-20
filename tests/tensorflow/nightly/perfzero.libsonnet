local common = import "common.libsonnet";
local timeouts = import "templates/timeouts.libsonnet";
local tpus = import "templates/tpus.libsonnet";
local gpus = import "templates/gpus.libsonnet";

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
    ],

    jobSpec+:: {
      template+: {
        spec+: {
          # HACK: remove publisher container
          initContainerMap: { },
        },
      },
    },
  },
  local resnet50KerasBenchmarkSynth = PerfZeroTest {
    modelName: "resnet50",
    benchmarkOptions+:: {
      test+: {
        module: "official.benchmark.keras_imagenet_benchmark",
        class: "Resnet50KerasBenchmarkSynth",
      }
    },
  },

  local v100 = resnet50KerasBenchmarkSynth {
    accelerator: gpus.teslaV100,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_1_gpu_no_dist_strat",
      },
    },
  },
  local v3_8 = resnet50KerasBenchmarkSynth {
    accelerator: tpus.v3_8,
    benchmarkOptions+:: {
      test+: {
        method: "benchmark_2x2_tpu_bf16",
      },
    },
  },

  configs: [
    resnet50KerasBenchmarkSynth + v100,
    resnet50KerasBenchmarkSynth + v3_8,
  ],
}