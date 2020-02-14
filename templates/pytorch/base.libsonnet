local base = import "../base.libsonnet";

{
  PyTorchTest:: base.BaseTest {

    regressionTestConfig+: {
      metric_subset_to_alert: [
        "ExecuteTime__Percentile_99_sec_final",
	"CompileTime__Percentile_99_sec_final",
	"total_wall_time",
	"Accuracy/test_final",
      ],
      metric_success_conditions+: {
        "ExecuteTime__Percentile_99_sec_final": {
	  success_threshold: {
            stddevs_from_mean: 5.0,
	  },
	  comparison: "less",
	  wait_for_n_points_of_history: 10,
	},
        "CompileTime__Percentile_99_sec_final": {
	  success_threshold: {
            stddevs_from_mean: 5.0,
	  },
	  comparison: "less",
	  wait_for_n_points_of_history: 10,
	},
      },
    },

    metricCollectionConfig+: {
      "tags_to_ignore": ["LearningRate"]
    },

    image: "gcr.io/xl-ml-test/pytorch-xla",

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              args+: [ "--logdir=$(MODEL_DIR)" ],
              envMap+: {
                XLA_USE_BF16: "0",
              },
              resources+: {
                requests+: {
                  cpu: "4.5",
                  memory: "8Gi",
                },
              },

              volumeMounts: [{
                mountPath: "/dev/shm",
                name: "dshm",
              }],
            },
          },
          volumes: [
            {
              name: "dshm",
              emptyDir: {
                medium: "Memory",
              },
            },
          ],
        },
      },
    },
  },
}
