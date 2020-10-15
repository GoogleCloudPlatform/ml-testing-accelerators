local base = import 'templates/base.libsonnet';
local volumes = import "templates/volumes.libsonnet";

{
  PytorchTest:: base.BaseTest {
    frameworkPrefix: 'pt-1.4', # TODO: Change to match your version.
    image: 'gcr.io/xl-ml-test/pytorch-examples-gpu', # TODO: Change to your image.
    imageTag: 'nightly',  # TODO: Change to your image tag.
    publisherImage: 'gcr.io/xl-ml-test/publisher:latest',  # TODO: Change to your image.
    mode: 'conv',

    volumeMap+: {
      dshm: volumes.MemoryVolumeSpec {
        name: "dshm",
        mountPath: "/dev/shm",
      },
    },

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

  datasetsVolume: volumes.PersistentVolumeSpec {
    name: "pytorch-datasets-claim",
    mountPath: "/datasets",
  },
}
