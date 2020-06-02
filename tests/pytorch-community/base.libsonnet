local base = import '../../templates/base.libsonnet';
local gpus = import '../../templates/gpus.libsonnet';

{
  PytorchTest:: base.BaseTest {
    frameworkPrefix: 'pt-1.4', # TODO: Change to match your version.
    image: 'gcr.io/xl-ml-test/pytorch-community', # Change to your image.
    imageTag: 'latest',
    publisherImage: 'gcr.io/xl-ml-test/publisher:latest',
    mode: 'convergence',
    accelerator: gpus.teslaV100,

    regressionTestConfig: {
      metric_success_conditions: {
        "default": {
          "comparison": "less",
          "success_threshold": {
            "stddevs_from_mean": 4.0
          },
          "wait_for_n_points_of_history": 10
        },
      },
    },

    metricCollectionConfig: {
      "write_to_bigquery": true,
      "default_aggregation_strategies": ["final"]
    },
  },
}
