# Creating Automated Model Tests with JSonnet Templates

## Prerequisites

1. Complete all of the steps in the [previous tutorial](test-templates-basic.md).
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
1. Enable the [BiqQuery](console.cloud.google.com/apis/api/bigquery.googleapis.com) and [Stackdriver Error Reporting](console.cloud.google.com/apis/api/clouderrorreporting.googleapis.com) APIs.

## Automating tests with `CronJobs`

In the previous tutorial, you created a simple one-off Kubernetes `Job` to run the TensorFlow MNIST example model. In this tutorial, you will automate this `Job` to run every night and monitor performance metrics. You should have downloaded the MNIST dataset into a GCS bucket and created the `gcs-buckets` `ConfigMap` as described in the previous tutorial. Additionally, the file `mnist-cpu.jsonnet` should have the following contents:

```jsonnet
local base = import 'templates/base.libsonnet';

local mnist = base.BaseTest {
  frameworkPrefix: "tf",
  modelName: "mnist",
  mode: "example",

  timeout: 3600, # 1 hour, in seconds

  image: 'gcr.io/your-project/tensorflow', # Change 'your-project' to your project id
  imageTag: 'r2.2',

  accelerator: base.cpu,

  command: [
    'python3',
    'official/vision/image_classification/mnist_main.py',
    '--data_dir=$(MNIST_DIR)',
    '--num_gpus=0',
    '--train_epochs=1',
    '--model_dir=$(OUTPUT_BUCKET)/mnist/$(JOB_NAME)',
  ],
};

mnist.oneshotJob
```

Automated tests work by collecting [TensorBoard](https://www.tensorflow.org/tensorboard) metrics from model runs and automatically making assertions against them. This is done asynchronously in [Cloud Function](https://cloud.google.com/functions) called the Metrics Handler. When automated tests are triggered, they publish a message into a [PubSub](https://cloud.google.com/pubsub) queue with the location of the TensorBoard summaries, which metrics to collect, and how to aggregate them. This step is done as an [`initContainer`](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) in the pod template. These messages are then handled in batches at periodic intervals. The Metrics Handler aggregates the metrics according the configuration, and optionally adds special metrics not included in the TensorBoard summaries, `total_wall_time` and `time_to_accuracy`.

To build the init container, run the following command:


```bash
gcloud builds submit --config images/publisher/cloudbuild.yaml
```

Then, to deploy the metrics handler, run these commands:

```bash
# Deploy the Cloud Function and create the PubSub topic to trigger the function.
cd metrics_handler
gcloud functions deploy metrics_handler --runtime python37 --trigger-topic=begin-metrics-handler --memory=1024MB --entry-point=run_main --timeout=500s
cd -
# Create the PubSub queue for the Cloud Function to read from.
gcloud pubsub topics create metrics-written
# Trigger the Metrics Handler every 15 minutes.
gcloud scheduler jobs create pubsub metrics-handler --schedule="*/15 * * * *" --topic=begin-metrics-handler --message-body="{}" --description="Kicks off the metric handler"
```

Note that you can adjust the timing of the metrics handler by changing the `--schedule` argument in the last command above. This flag takes a schedule in [Cron format](https://en.wikipedia.org/wiki/Cron#CRON_expression). You can find the details of the new function on the [Cloud Functions](https://console.cloud.google.com/functions/list) page.

In `mnist-cpu.jsonnet`, add the `schedule` and `publisherImage` fields to the test definition. `schedule` is also in cron format, and `publisherImage` should be the `gcr.io` tag of the image you just built. Finally, replace `mnist.oneshotJob` with `mnist.cronJob`, and wrap the output in `std.manifestYamlDoc` to make the output more readable:

```jsonnet
local mnist = base.BaseTest {
  ...
  schedule: '0 */1 * * *',
  publisherImage: 'gcr.io/your-project/publisher:latest',
  ...
};

std.manifestYamlDoc(mnist.cronJob)
```

This will run the MNIST cpu training job at the top of every hour. Build the template with JSonnet:

```bash
jsonnet -S -J ml-testing-accelerators/ mnist-cpu.jsonnet
```

Note the following changes to the output:

1. The `Job` template is now embedded in a `CronJob` resource.
1. The `Pod` template now has an `initContainer`, `publisher`, that publishes a message to the PubSub queue.
1. The `METRIC_CONFIG` environment variable of `publisher` automatically tracks the `total_wall_time` of the Job and asserts that it should not deviate by more than 5 standard deviations from the mean.

This `CronJob` will automatically run and its metrics will be periodically collected by the Metrics Handler. Note that, by default, all TensorBoard summaries are collected by the Metrics Handler, but only cerain metrics have assertions.

## Configuring metric aggregation

You can configure metric aggregation and regression alerting with the `metricCollectionConfig` and `regressionTestConfig` field of `BaseTest`, respectively. For this example, we can set a target final accuracy of 98% for the test:

```
local mnist = base.BaseTest {
  ...
  metricCollectionConfig+: {
    metric_to_aggregation_strategy: {
      epoch_sparse_categorical_accuracy: ['final'],
    },
  },
  regressionTestConfig: {
    metric_success_conditions: {
      epoch_sparse_categorical_accuracy_final: {
        comparison: 'greater',
        success_threshold: {
          fixed_value: 0.96,
        },
      },
    },
  },
};
```

If you rebuild the template, you can see that this config has updated the `METRIC_CONFIG` environment variable of publisher. You can deploy the `CronJob` resource with the following command:

```bash
jsonnet -S -J ml-testing-accelerators/ mnist-cpu.jsonnet | kubectl apply -f -
```

You can find deployed `CronJob` running on your [GKE workloads](https://console.cloud.google.com/kubernetes/workload) page. You can either wait for the `CronJob` trigger, or manually trigger a run of the cronjob with the following command:

```
kubectl create job --from=cronjob/tf-mnist-example-cpu tf-mnist-example-cpu-$(date +"%Y%m%d%H%M")
```

If you look at the beginning of the logs for your job, you'll see a message like the following from the `publisher` container:

```
messageIds:
- '1234567890987654321'
```

Once the job finishes, wait for the next trigger of the Metrics Handler or trigger the `metrics-handler` job from the [Cloud Scheduler](console.cloud.google.com/cloudscheduler) page in your console. From the `metrics_handler` [Cloud Function](https://pantheon.corp.google.com/functions/list) page, follow the link to view logs. You should see a message like the following:

```
Processed a message for each of the following tests: ['tf-example-mnist-cpu']
```

If you go further up in the logs, you'll find a warning that the job failed a metrics assertion! For example:

```
Metric `epoch_sparse_categorical_accuracy_final` was out of bounds for test `tf-mnist-example-cpu`. Bounds were (0.98, inf) and value was 0.79
```

That's because we set the accuracy threshold at 96%, but only ran the job for one epoch. To fix the issue, update the `--train_epochs` flag in the test:

```
local mnist = base.BaseTest {
  ...
  command: [
    ...
    '--train_epochs=10',
    ...
  ],
};
```

Then, manually run the job again with the same command. This run will take longer, since it will run multiple epochs without an accelerator. Once it is completed, check the metrics handler logs and note that no warning has been generated. You can also check the job status history in [BiqQuery](https://console.cloud.google.com/bigquery) with the following query:

```sql
SELECT *
FROM metrics_handler_dataset.job_history
```

You can also view the model's metric history for the model with the following query:

```sql
SELECT *
FROM metrics_handler_dataset.metric_history
```

This tutorial runs a `Job` with no accelerator, but you can add GPUs or TPUs as in the previous tutorial.

## Next Steps

1. Configure a [dashboard](../../dashboard/README.md) to view your metric status over time.
1. Configure [e-mail alerts](../../metrics_handler/README.md) for the Metrics Handler.
