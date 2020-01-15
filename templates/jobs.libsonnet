local timeouts = import "timeouts.libsonnet";
local tpus = import "tpus.libsonnet";

{
  JobConfig:: {
    local config = self,

    framework_prefix: error "Must specify `framework_prefix`",
    model_name: error "Must specify `model_name`",
    accelerator: tpus.v2_8 + tpus.Preemptible,
    # HACK: for format strings
    accelerator_name:: config.accelerator.name,
    mode: "functional",
    command: error "Must specify model `command`",
    framework_version: error "Must specify `framework_version`",
    image: error "Must specify mode `image`",
    image_tag: 'latest',
    timeout: error "Must specify `timeout`", # 1 hour

    # TODO: give these reasonable defaults
    metric_collection_config: null,
    regression_test_config: null,

    job_name:: '%(framework_prefix)s-%(model_name)s-%(mode)s-%(accelerator_name)s' % config,
    job_spec:: {
      # Try 3 times before giving up
      backoffLimit: 2,
      activeDeadlineSeconds: config.timeout,
      template: {
        metadata: {
          annotations: {
            'tf-version.cloud-tpus.google.com': config.framework_version,
          },
        },
        spec: {
          restartPolicy: 'Never',
          containers: [
            {
              name: config.job_name,
              image: '%(image)s:%(image_tag)s' % config,
	            # Use Docker image's entrypoint wrapper
              args: config.command,
              resources: {
                limits: config.accelerator.resource_limits,
              },
              env: [
                {
                  name: 'POD_NAME',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: 'metadata.name'
                    },
                  },
                },
                {
                  name: 'POD_UID',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: 'metadata.uid'
                    },
                  },
                },
                {
                  name: 'POD_NAMESPACE',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: 'metadata.namespace'
                    },
                  },
                },
                {
                  name: 'JOB_NAME',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: "metadata.labels['job-name']",
                    },
                  },
                },
                {
                  name: 'MODEL_DIR',
                  # TODO: Factor output bucket out into a ConfigMap
                  value: 'gs://xl-ml-test-us-central1/k8s/%(model_name)s/%(mode)s/%(accelerator_name)s/$(JOB_NAME)' % config,
                },
                {
                  name: 'METRIC_COLLECTION_CONFIG',
                  value: std.base64(std.manifestJsonEx(config.metric_collection_config, '  '))
                },
                {
                  name: 'REGRESSION_TEST_CONFIG',
                  value: std.base64(std.manifestJsonEx(config.regression_test_config, '  '))
                },
              ],
            },
          ],
        },
      },
    },

    cron_job(schedule):: {
      apiVersion: 'batch/v1beta1',
      kind: 'CronJob',
      metadata: {
        name: config.job_name,
      },
      spec: {
        schedule: schedule,
        concurrencyPolicy: 'Forbid',
        jobTemplate: {
          spec: config.job_spec,
        },
      },
    },
  },
}
