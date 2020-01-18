local jobs = import '../jobs.libsonnet';

{
  PyTorchJobConfig:: jobs.JobConfig {

    regression_test_config+: {
      "threshold_expression_overrides": {
        "Accuracy/test_final": "v_mean - (v_stddev * 3.0)"
      },
      "comparison_overrides": {
        "Accuracy/test_final": "COMPARISON_LT"
      }
    },

    image: 'gcr.io/xl-ml-test/pytorch-xla',
    job_spec+:: {
      template+: {
        spec+: {
          volumes: [{
            name: 'dshm',
            emptyDir: {
              medium: 'Memory',
            },
          }],
          containers: [
            container {
              args+: ['--logdir=$(MODEL_DIR)' ],
              volumeMounts: [{
                mountPath: '/dev/shm',
                name: 'dshm',
              }],
              env+: [{
                name: 'XLA_USE_BF16',
                value: '0',
              }],
              resources+: {
                requests+: {
                  cpu: '2.5',
                  memory: '8Gi',
                },
              },
            } for container in super.containers
          ],
        },
      },
    },
  },
}
