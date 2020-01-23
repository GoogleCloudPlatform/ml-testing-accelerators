local base = import '../base.libsonnet';

{
  GardenTest:: base.BaseTest {
    local config = self,

    regression_test_config+: {
      "threshold_expression_overrides": {
        "epoch_sparse_categorical_accuracy_final": "v_mean - (v_stddev * 3.0)"
      },
      "comparison_overrides": {
        "epoch_sparse_categorical_accuracy_final": "COMPARISON_LT"
      }
    },

    image: 'gcr.io/xl-ml-test/model-garden',
    job_spec+:: {
      template+: {
        spec+: {
          containers: [
            container {
              args+: ['--model_dir=$(MODEL_DIR)' ],
            } for container in super.containers
          ],
        },
      },
    },
  }
}
