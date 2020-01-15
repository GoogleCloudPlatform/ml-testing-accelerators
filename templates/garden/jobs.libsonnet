local jobs = import '../jobs.libsonnet';

{
  GardenJobConfig:: jobs.JobConfig {
    local config = self,

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
