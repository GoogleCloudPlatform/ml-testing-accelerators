local jobs = import '../jobs.libsonnet';

{
  GardenJobConfig:: jobs.JobConfig {
    local config = self,

    image: 'gcr.io/wcromar-tpus/model-garden',
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
