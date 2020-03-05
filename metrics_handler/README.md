# Metrics Handler

The metrics handler runs as a Cloud Function that is kicked off periodically by a Cloud Scheduler.

Each time it runs, it checks for tests that have finished running and collects the final status of each test plus any training metrics that the job wrote to Tensorboard. It writes the job status into a BigQuery table called `job_history` and writes the metrics into a BigQuery table called `metric_history`.


## Setup

Once you have cloned the repo, create the Cloud Function + Cloud Scheduler with
these commands:

1. `cd metrics_handler`
2. `gcloud functions deploy metrics_handler --runtime python37 --trigger-topic=begin-metrics-handler --memory=1024MB --entry-point=run_main --timeout=500s`
3. `gcloud scheduler jobs create pubsub metrics-handler --schedule="*/15 * * * *" --topic=begin-metrics-handler --message-body="{}" --description="Kicks off the metric handler"`


## Config

The behavior of the metrics handler is configurable on a per-test basis. See [examples/metrics](examples/metrics) for some example config files, which are written in JSON. These are loaded into your test config via the `METRIC_CONFIG` key as shown in [examples/pt-cifar-tpu-v3-8.yaml](examples/pt-cifar-tpu-v3-8.yaml)

There are 2 areas to configure:
1. `metric_collection_config` controls which training metrics are collected when your test runs and how those metrics are aggregated.
2. `regression_test_config` controls alerting and error reporting for each metric.


#### metric_collection_config

By default, the metrics handler will collect all metrics that your test writes to Tensorboard and will aggregate by using the final value for each metric. It will write these values to BigQuery.

Full config options:

```
"metric_collection_config": {
  # (Required) How to choose the value of each metric written to BigQuery.
  # For each strategy, one row will be written per test run and the metric
  # name will be e.g. accuracy_final or accuracy_max. Available strategies:
  # "final": Save the final value for that metric.
  # "max": Save the max value for that metric.
  # "min": Save the min value for that metric.
  "default_aggregation_strategies": ["final"],

  # (Optional) Apply special aggregation strategies to some metrics. Key
  # is the Tensorboard tag of the metric that your model writes during
  # training and value is a list of strategies as described above.
  "metric_to_aggregation_strategy": {
    "loss": ["final", "min"],  # Save final loss and min loss.
    "accuracy": ["max"],       # Save max accuracy only.
  }

  # (Optional) Ignore metrics with these Tensorboard tags. These metrics will
  # not be aggregated or written to Bigquery and the metrics handler will not
  # look for regressions in these metrics.
  "tags_to_ignore": ["LearningRate"],

  # (Optional) Compute the wall time between the test starting and the value
  # of a specified Tensorboard tag reaching a specified threshold. If set,
  # the metrics handler will write a row with metric name `time_to_accuracy`
  # for each test run or send an alert if the threshold is never reached.
  "time_to_accuracy": {
    "accuracy_threshold": 76.0,  # Compute time needed to reach 76% acc.
    "accuracy_tag": "Accuracy/test",  # Tag used in the SummaryWriter for acc.
  },

  # (Optional) Defaults to True. Set to false to disable all Bigquery writes.
  "write_to_bigquery": "True",
}
```


#### regression_test_config

The metrics handler can send alerts if any training metrics regress. These
alerts will appear in Stackdriver Error Reporting for your project, where you
can ack, resolve, or link incidents to bugs. See [here](https://cloud.google.com/error-reporting/docs/notifications) for how to set up notifications for incidents.

By default, no alerts will fire for regressions in training metrics but they will fire if any test crashes or times out.

Full config options:
```
"regression_test_config": {
  # (Required) Defines the success condition for each training metric. If any
  # metric does not meet the success condition, it will trigger an alert.
  #
  # Key for each entry is the metric name. Note that most metric names will
  # take the form of "<tensorboard tag>_<aggregration strategy>". Use the
  # special key "default" to set a success condition that will be used for any
  # metric that does not have an explicit success condition.
  #
  # NOTE: All metrics captured by metric_success_conditions will compute
  # lower/upper bounds and write them to BigQuery so that these bounds
  # can be used as a visual aid when viewing metric history. You have the
  # option of disabling alerts for any or all of these metrics with the other
  # optional fields described below.
  "metric_success_conditions": {
    "Accuracy/test_final": {
      "comparison": "greater",
      "success_threshold": {
        # Compare the metric value to a fixed value. For this type of
        # threshold, comparison can be "greater", "less", or "equal".
        "fixed_value": 99.0
      }
    },
    "default": {
      "comparison": "less",
      "success_threshold": {
        # Compare the metric to historical performance. For this type
        # of threshold, comparison can be "greater" or "less".
        # "greater": assert metric_value > (mean - 4.0 * stddev)
        # "less": assert metric_value < (mean + 4.0 * stddev)
        "stddevs_from_mean": 4.0
      },
      # (Optional) Wait for N points of data before enforcing this success
      # condition so that mean/stddev are more reliable. If not set, alerts
      # will probably fire on the first run since the standard deviation
      # will be 0.0.
      "wait_for_n_points_of_history": 10
    }
  }

  # (Optional) If set, only these metric names will create alerts if their
  # success condition fails.
  "metric_subset_to_alert": [
    "total_wall_time",
    "Accuracy/test_final"
  ],

  # (Optional) Defaults to True. Set to False to disable Stackdriver alerting
  # for training metrics of this test.
  "write_to_error_reporting": "True",
}
```
