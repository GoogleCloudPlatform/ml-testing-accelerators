local base = import "../base.libsonnet";

{
  GardenTest:: base.BaseTest {
    local config = self,

    regressionTestConfig+: {
      threshold_expression_overrides: {
        "epoch_sparse_categorical_accuracy_final": "v_mean - (v_stddev * 3.0)"
      },
      comparison_overrides: {
        "epoch_sparse_categorical_accuracy_final": "COMPARISON_LT"
      }
    },

    image: "gcr.io/xl-ml-test/model-garden",

    jobSpec+:: {
      template+: {
        spec+: {
          containerMap+: {
            train+: {
              args+: [ "--model_dir=$(MODEL_DIR)" ]
            },
          },
        },
      },
    },
  },
}
