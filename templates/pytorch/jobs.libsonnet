local jobs = import '../jobs.libsonnet';

{
  PyTorchJobConfig:: jobs.JobConfig {
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