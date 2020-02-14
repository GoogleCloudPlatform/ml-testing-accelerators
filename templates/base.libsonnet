local timeouts = import "timeouts.libsonnet";
local tpus = import "tpus.libsonnet";

{
  BaseTest:: {
    local config = self,

    frameworkPrefix: error "Must specify `frameworkPrefix`",
    modelName: error "Must specify `modelName`",
    accelerator: error "Must specify `accelerator`",
    # HACK: for format strings
    acceleratorName:: config.accelerator.name,
    mode: "functional",
    command: error "Must specify model `command`",
    tpuVersion: error "Must specify `tpuVersion`",
    image: error "Must specify mode `image`",
    imageTag: "latest",
    timeout: error "Must specify `timeout`", # 1 hour
    # Schedule for CronJob in UTC
    schedule: error "Must specify `schedule`",

    metricCollectionConfig: {
      write_to_bigquery: true,
      default_aggregation_strategies: ["final"],
    },
    regressionTestConfig: {
      write_to_error_reporting: true,
      metric_success_conditions: {
	"total_wall_time": {
	  success_threshold: {
            stddevs_from_mean: 5.0,
	  },
	  comparison: "less",
	  wait_for_n_points_of_history: 10,
	},
      },
    },

    testName:: "%(frameworkPrefix)s-%(modelName)s-%(mode)s-%(acceleratorName)s" % config,
    jobSpec:: {
      # Try 3 times before giving up
      backoffLimit: 2,
      activeDeadlineSeconds: config.timeout,
      template: {
        metadata: {
          annotations: {
            "tf-version.cloud-tpus.google.com": config.tpuVersion,
          },
        },
        spec: config.accelerator.PodSpec {
          local pod = self,

          restartPolicy: "Never",
          containerMap+:: {
            train+: {
              local main = self,

              image: "%(image)s:%(imageTag)s" % config,
              imagePullPolicy: "Always",
              # Use Docker image's entrypoint wrapper
              args: config.command,

              envMap:: {
                TEST_NAME: config.testName,
                MODEL_DIR:
                  "gs://xl-ml-test-us-central1/k8s/%(modelName)s/%(mode)s/%(acceleratorName)s/$(JOB_NAME)" % config,
                METRIC_COLLECTION_CONFIG:
                  std.manifestJsonEx(config.metricCollectionConfig, " "),
                REGRESSION_TEST_CONFIG:
                  std.manifestJsonEx(config.regressionTestConfig, " "),
              },
              env: [
                {
                  name: "POD_NAME",
                  valueFrom: {
                    fieldRef: {
                      fieldPath: "metadata.name"
                    },
                  },
                },
                {
                  name: "POD_UID",
                  valueFrom: {
                    fieldRef: {
                      fieldPath: "metadata.uid"
                    },
                  },
                },
                {
                  name: "POD_NAMESPACE",
                  valueFrom: {
                    fieldRef: {
                      fieldPath: "metadata.namespace"
                    },
                  },
                },
                {
                  name: "JOB_NAME",
                  valueFrom: {
                    fieldRef: {
                      fieldPath: "metadata.labels['job-name']",
                    },
                  },
                },
              ] + [
                {
                  name: key,
                  value: main.envMap[key],
                }
                for key in std.objectFields(main.envMap)
              ],
            },
          },
          containers: [
            { name: name } + pod.containerMap[name]
              for name in std.objectFields(pod.containerMap)
          ],
        },
      },
    },

    oneshotJob:: {
      // Don't record metrics from oneshot jobs
      local oneshotConfig = config {
        metricCollectionConfig+: {
          write_to_bigquery: false,
        },
        regressionTestConfig: null,
      },

      apiVersion: "batch/v1",
      kind: "Job",
      metadata: {
        generateName: "%s-" % oneshotConfig.testName,
      },
      spec: oneshotConfig.jobSpec,
    },

    cronJob:: {
      apiVersion: "batch/v1beta1",
      kind: "CronJob",
      metadata: {
        name: config.testName,
        namespace: "automated"
      },
      spec: {
        schedule: config.schedule,
        concurrencyPolicy: "Forbid",
        jobTemplate: {
          spec: config.jobSpec,
        },
      },
    },
  },
}
