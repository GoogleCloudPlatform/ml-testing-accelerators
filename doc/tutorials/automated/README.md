# Creating Automated Model Tests with JSonnet Templates

## Prerequisites

It is strongly recommended that you work through the [`basic` tutorial](../basic) first, since we will be building off of concepts from there.

1. A copy of this repository.
1. Dependencies from [developing.md](/doc/developing.md).
1. You do not need accelerators enabled to follow this tutorial. If you deleted your cluster from the previous tutorial, you may run the following to create a new one:
    ```bash
    gcloud beta container clusters create tutorial-cluster \
    --zone us-central1-b \
    --release-channel regular \
    --machine-type n1-standard-4 \
    --scopes "https://www.googleapis.com/auth/cloud-platform" \
    --num-nodes 1 \
    --enable-ip-alias \
    --enable-autoupgrade \
    --project=$PROJECT_ID
    ```
    ```bash
    # Connect to your cluster
    gcloud container clusters get-credentials tutorial-cluster --project $PROJECT_ID --zone us-central1-b
    ```
1. A GCS bucket, if you didn't create one in the previous tutorial.
  - To create a new GCS bucket, run `gsutil mb -c standard -l us-central1 gs://your-bucket-name`

Before you begin, set the following environment variables:

```bash
$ export PROJECT_ID=...
$ export GCS_BUCKET=gs://...
```

Log in with `gcloud`

```bash
$ gcloud auth login --update-adc
```

## Introduction and setup

In the previous tutorial, you created a simple one-off Kubernetes `Job` to run the TensorFlow MNIST example model. In this tutorial, you will automate this `Job` to run every night and monitor performance metrics.

Automated tests work by collecting [TensorBoard](https://www.tensorflow.org/tensorboard) metrics from model runs and checking assertions against them. Each cluster and namespace that has automated tests runs a process called the Event Publisher that watches for completed jobs and publishes a message to a [Cloud PubSub](https://cloud.google.com/pubsub) topic for each one. Asynchronously, a [Cloud Function](https://cloud.google.com/functions) called the Metrics Handler triggers on each message and collects any TensorBoard metrics written to GCS, computes metric bounds for the test based on historical data, and records the test results in BigQuery.

To get started, create a pubsub topic for test results:

```bash
$ gcloud pubsub topics create tutorial-topic --project=$PROJECT_ID
```

Next, create a BigQuery dataset to permanently store test data:

```bash
$ cd metrics
metrics$ bazel run handler:create_bq_tables -- --project=$PROJECT_ID --dataset=tutorial_dataset
```

Create the Metrics Handler:

```bash
metrics$ bazel run handler:deploy -- --name tutorial-handler --topic tutorial-topic --dataset tutorial_dataset --project $PROJECT_ID
```

Build and deploy the Event Publisher:

```bash
metrics$ bazel run --define project=$PROJECT_ID //publisher:image_push
```

You should see a message like `Successfully pushed Docker image to gcr.io/$PROJECT_ID/event-publisher:$USER_1655938667` -- save the image name for the next step:

```bash
export PUBLISHER_IMAGE_URL=gcr.io/...
```

Use `envsubst` to template in the environment variables for your project, and then deploy the Event Publisher to your cluster:

```bash
metrics$ PUBSUB_TOPIC=tutorial-topic envsubst '$PUBLISHER_IMAGE_URL $GCS_BUCKET $PUBSUB_TOPIC' < publisher/deployment/event-publisher.yaml | kubectl apply -f -
```

Double check that the Event Publisher is now running in your cluster:

```bash
$ kubectl get deployments
```

## Automating tests

Compare [`mnist-cpu.jsonnet`](../basic/mnist-cpu.jsonnet) to [`mnist-cpu-cronjob.jsonnet`](mnist-cpu-automated.jsonnet). To run this test automatically on a schedule, we have to make two changes:


```jsonnet
local mnist = base.BaseTest {
  [...]
  // Run every hour, on the hour
  schedule: '0 */1 * * *',
  [...]
};

std.manifestYamlDoc(mnist.cronJob, quote_keys=false)
```

First, we have to set `schedule` in a `cron` format, and then we change the output to use `cronJob` instead of `oneshotJob`. Build with `jsonnet` and note the differences in the generated output:

```bash
# Run from the root of this repository
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/automated/mnist-cpu-cronjob.jsonnet
```

Instead of a `Job` resource, the output is now a `CronJob` that runs according to the `schedule` set in `mnist-cpu-cronjob.jsonnet`:

```bash
apiVersion: "batch/v1"
kind: "CronJob"
metadata:
  labels:
    [...]
    benchmarkId: "tf-mnist-example-cpu"
    [...]
spec:
  jobTemplate:
    metadata:
      labels:
        [...]
        benchmarkId: "tf-mnist-example-cpu"
        [...]
  [...]
  schedule: "0 */1 * * *"
```

Also note the `benchmarkId` field in `labels` in both the `CronJob` metadata and the `jobTemplate` metadata. The Event Publisher watches for `Job`s in the same namespace (`default` in our case) that have the `benchmarkId` metadata field set to identify tests. When it sees a test complete, it publishes a message to the Metrics Handler, which records the test status in BigQuery.

Create and trigger a run of this test:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/automated/mnist-cpu-cronjob.jsonnet | kubectl apply -f -
# You can also find CronJob in Cloud Console and click "Run Now"
$ kubectl create job --from=cronjob.batch/tf-mnist-example-cpu tf-mnist-example-cpu-manual
```

Watch the job and wait for it to complete successfully:

```bash
$ kubectl logs job.batch/tf-mnist-example-cpu-manual -f
```

You should see a message like the following to indicate that a PubSub message has been published:

```bash
I0623 16:50:51.947220 140470884943680 __main__.py:202] Published message for tf-mnist-example-cpu-manual:
benchmark_id: "tf-mnist-example-cpu"
output_path: "gs://wcromar-tmp/tutorial2022/tf-mnist-example-cpu-manual"
num_attempts: 1
start_time {
  seconds: 1656002806
}
duration {
  seconds: 245
}
[...]
metric_collection_config {
}
[...]
labels {
  key: "benchmarkId"
  value: "tf-mnist-example-cpu"
}
[...]
```

Next, check the logs for the Metrics Handler:

```bash
gcloud functions logs read tutorial-handler
```

You should see a message like this, indicating that the job status was written to BigQuery:

```bash
INFO:absl:Inserting 1 rows into BigQuery table `$PROJECT_ID.tutorial_dataset.job_history`
```

Finally, view your job's status in BigQuery using `bq` or Cloud Console:

```bash
$ bq --project_id=$PROJECT_ID query "SELECT test_name, job_status, timestamp FROM tutorial_dataset.job_history"
```

You should see the status for the test run you triggered in from the `CronJob`:

```
+----------------------+------------+---------------------+
|      test_name       | job_status |      timestamp      |
+----------------------+------------+---------------------+
| tf-mnist-example-cpu | success    | 2022-06-23 16:46:46 |
+----------------------+------------+---------------------+
```

## Collecting metrics

By default, the Metrics Handler does not collect output metrics. In this section, we will configure metric collection for our example test to 1) record the final values of all TensorBoard metrics to BigQuery and 2) assert that the duration of the test does not increase beyond 2 standard deviations from the historical mean. Metrics are configured by populating `metricConfig` with an instance of `MetricCollectionConfig`, as defined in [`metrics.proto`](../../../metrics/metrics.proto). Note the changes to `metricConfig` in [`mnist-cpu-cronjob-metrics.jsonnet`](mnist-cpu-cronjob-metrics.jsonnet)

```jsonnet
local metrics = import 'templates/metrics.libsonnet';

local mnist = base.BaseTest {
  [...]

  metricConfig: metrics.MetricCollectionConfigHelper {
    sourceMap:: {
      tensorboard: metrics.TensorBoardSourceHelper {
        include_tags: [
          {
            tag_pattern: '*',
            strategies: [
              'FINAL',
            ],
          },
        ],
      },
      literals: {
        assertions: {
          duration: {
            std_devs_from_mean: {
              comparison: 'LESS',
              std_devs: 2,
            },
          },
        },
      },
    },
  },
};
```

`MetricCollectionConfigHelper` is an optional template to help build instances of `MetricCollectionConfig`. Likewise, `TensorBoardSourceHelper` is a template to help build instances of `TensorBoardSource`. In this example, we collect metrics from two sources:

- TensorBoard: Models record multiple values for each "tag" (e.g. `train/epoch_loss`). We have to tell the Metrics handler how to aggregate those multiple values into one value (i.e. the aggregation "strategy"). In this example specifically, we want to collect the _final_ value of each TensorBoard metric. We want to match every tag, so we use `tag_pattern: *`. If you wanted to collect only training metrics, your `tag_pattern` might be `train/*`, depending on the format of your model's TensorBoard metric names.
- Test duration: `duration` is a special value that is computed from a `Job`'s start and end times. In this example, we also set up an assertion that the `duration` (in seconds) must not rise more than two standard deviations from the historical mean.

You can similarly set up assertions for TensorBoard metrics as well in `TensorBoardSource.aggregate_assertions`. For a full list of assertion types (e.g. percent difference from mean or comparison to a fixed value), aggregation strategies (e.g. maximum or average), and comparisons (e.g. GREATER or EQUAL), see `metrics.proto`.

Build the updated `CronJob` and compare the output:

```
[...]
spec:
  concurrencyPolicy: "Forbid"
  jobTemplate:
    metadata:
      annotations:
        ml-testing-accelerators/gcs-subdir: "tf/mnist/example/cpu"
        ml-testing-accelerators/metric-config: |
          {
            "sources": [
              {
                "literals": {
                  "assertions": {
                    "duration": {
                      "std_devs_from_mean": {
                        "comparison": "LESS",
                        "std_devs": 2
                      }
                    }
                  }
                }
              },
              {
                "tensorboard": {
                  "aggregate_assertions": [

                  ],
                  "include_tags": [
                    {
                      "strategies": [
                        "FINAL"
                      ],
                      "tag_pattern": "*"
                    }
                  ]
                }
              }
            ]
          }
[...]
```

Note how the `ml-testing-accelerators/metric-config` field corresponds to the `metricConfig` template field, except `sourceMap` is turned into the `sources` list. `tensorboard.aggregate_assertions` is empty because we did not define any assertions for TensorBoard metrics Deploy the updated `CronJob` and trigger a run:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/automated/mnist-cpu-cronjob-metrics.jsonnet | kubectl apply -f -
$ kubectl create job --from=cronjob.batch/tf-mnist-example-cpu tf-mnist-example-cpu-manual-metrics
```

Once that job completes, check BigQuery to see the recorded metrics:

```bash
$ bq --project_id=$PROJECT_ID query "SELECT test_name, metric_name, metric_value, timestamp FROM tutorial_dataset.metric_history"
```

You should see output like the following, including both `duration` and the TensorBoard output from the MNIST example model:

```
+----------------------+-----------------------------------------------------------------------+---------------------+---------------------+
|      test_name       |                              metric_name                              |    metric_value     |      timestamp      |
+----------------------+-----------------------------------------------------------------------+---------------------+---------------------+
| tf-mnist-example-cpu | train/epoch_sparse_categorical_accuracy_final                         |  0.5007240176200867 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | validation/evaluation_sparse_categorical_accuracy_vs_iterations_final |  0.7775607705116272 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | train/epoch_loss_final                                                |  1.5759683847427368 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | validation/evaluation_loss_vs_iterations_final                        |  0.6856215000152588 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | duration                                                              |               291.0 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | train/epoch_learning_rate_final                                       | 0.04999881610274315 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | validation/epoch_loss_final                                           |  0.6856215000152588 | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu | validation/epoch_sparse_categorical_accuracy_final                    |  0.7775607705116272 | 2022-06-23 17:26:12 |
+----------------------+-----------------------------------------------------------------------+---------------------+---------------------+
```

Note how the TensorBoard metrics have a suffix indicating the aggregation strategy that was used to collect the metric. In our example, we only collected the `FINAL` value, so all of them have the `_final` suffix.

For `duration` in particular, our example has an assertion based on the historical mean and standard deviation. You can see the bounds in the `metric_lower_bound` and `metric_upper_bound` columns:

```bash
$ bq --project_id=$PROJECT_ID query "SELECT test_name, metric_value, metric_lower_bound, metric_upper_bound, timestamp FROM tutorial_dataset.metric_history WHERE metric_name='duration'"
+----------------------+--------------+--------------------+--------------------+---------------------+
|      test_name       | metric_value | metric_lower_bound | metric_upper_bound |      timestamp      |
+----------------------+--------------+--------------------+--------------------+---------------------+
| tf-mnist-example-cpu |        291.0 |               NULL |               NULL | 2022-06-23 17:26:12 |
+----------------------+--------------+--------------------+--------------------+---------------------+
```

Because there are no previous data points, the mean and standard deviation are not defined, and there are no computed bounds. If you trigger the job two more times, the third run will have bounds set:

```bash
$ bq --project_id=$PROJECT_ID query "SELECT test_name, metric_value, metric_lower_bound, metric_upper_bound, timestamp FROM tutorial_dataset.metric_history WHERE metric_name='duration'"
Waiting on bqjob_r157b47e474061b95_0000018191be05d0_1 ... (0s) Current status: DONE
+----------------------+--------------+--------------------+--------------------+---------------------+
|      test_name       | metric_value | metric_lower_bound | metric_upper_bound |      timestamp      |
+----------------------+--------------+--------------------+--------------------+---------------------+
| tf-mnist-example-cpu |        291.0 |               NULL |               NULL | 2022-06-23 17:26:12 |
| tf-mnist-example-cpu |        262.0 |               NULL |               NULL | 2022-06-23 17:51:15 |
| tf-mnist-example-cpu |        267.0 |               NULL |              305.5 | 2022-06-23 18:00:00 |
+----------------------+--------------+--------------------+--------------------+---------------------+
```

## Next steps

- Create a dashboard to view the test data in BigQuery (e.g. with [Data Studio](https://datastudio.google.com/)).
- Look at `metrics.proto` to see all metric collection options.
- If your code does not already use TensorFlow and TensorBoard, look into [`tensorboardX`](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html) to write metrics.
