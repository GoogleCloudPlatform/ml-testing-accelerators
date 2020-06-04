# Metrics Handler

The metrics handler runs as a Cloud Function that is kicked off periodically by a Cloud Scheduler.

Each time it runs, it checks for tests that have finished running and collects the final status of each test plus any training metrics that the job wrote to Tensorboard. It writes the job status into a BigQuery table called `job_history` and writes the metrics into a BigQuery table called `metric_history`.


## Setup

Once you have cloned the repo, create the Cloud Function + Cloud Scheduler with these commands:

1. `cd metrics_handler`
2. `gcloud functions deploy metrics_handler --runtime python37 --trigger-topic=begin-metrics-handler --memory=1024MB --entry-point=run_main --timeout=500s`
3. `gcloud scheduler jobs create pubsub metrics-handler --schedule="*/15 * * * *" --topic=begin-metrics-handler --message-body="{}" --description="Kicks off the metric handler"`


### (Optional) Set up notification emails

After the basic setup above, you'll be able to see errors in Stackdriver Error Reporting and collect metrics such as job failures or accuracy of your models being tested. If you want better email notifications, follow these steps too:

1. Set up a [SendGrid account](https://sendgrid.com/free/). Currently, this service is free for up to 100 emails sent per day.
1. Create a SendGrid API key:
    * Log in to your SendGrid account.
    * Navigate to Settings > API Keys.
    * Create a new API key with full access.
    * Copy the API key when it is displayed.
1. Navigate to the [Cloud Secrets page](https://console.cloud.google.com/security/secret-manager) for your project.
1. Use the "Create Secret" button to create 3 Cloud Secrets:
    * `alert-destination-email-address`: Your alerts will be sent to this email.
    * `alert-sender-email-address`: The 'from' email used by Sendgrid for alerts.
    * `sendgrid-api-key`: Sendgrid API key that you copied in step 2.
1. Give permission to your metrics handler Cloud Function to read these 3 Secrets. Run these from command line:
    * `gcloud beta secrets add-iam-policy-binding alert-destination-email-address --role roles/secretmanager.secretAccessor --member serviceAccount:YOUR-PROJECT-NAME@appspot.gserviceaccount.com`
    * `gcloud beta secrets add-iam-policy-binding alert-sender-email-address --role roles/secretmanager.secretAccessor --member serviceAccount:YOUR-PROJECT-NAME@appspot.gserviceaccount.com`
    * `gcloud beta secrets add-iam-policy-binding sendgrid-api-key --role roles/secretmanager.secretAccessor --member serviceAccount:YOUR-PROJECT-NAME@appspot.gserviceaccount.com`

Once you've finished these steps, the metrics handler will send an alert email to the `alert-destination-email-address` whenever it finds errors in your tests.


## Config

The behavior of the metrics handler is configurable on a per-test basis. See [examples/metrics](../examples/metrics) for some example config files, which are written in JSON. These are loaded into your test config with the `METRIC_CONFIG` key as shown in [examples/pt-cifar-tpu-v3-8.yaml](../examples/pt-cifar-tpu-v3-8.yaml)

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
  # "average": Save the average value for that metric.
  "default_aggregation_strategies": ["final"],

  # (Optional) Apply special aggregation strategies to some metrics. Key
  # is the Tensorboard tag of the metric that your model writes during
  # training and value is a list of strategies as described above.
  "metric_to_aggregation_strategy": {
    "loss": ["final", "min"],  # Save final loss and min loss.
    "accuracy": ["max"],       # Save max accuracy only.
  }

  # (Optional) Ignore metrics with these Tensorboard tags. These metrics will
  # not be aggregated or written to BigQuery and the metrics handler will not
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

  # (Optional) Defaults to True. Set to false to disable all BigQuery writes.
  "write_to_bigquery": "True",

  # (Optional) Defaults to false. Set to true to prefix tag names with
  # the run name, or the subdirectory containing TensorBoard summaries (e.g.
  # `train/` or `eval/`).
  "use_run_name_prefix: false
}
```


#### regression_test_config

The metrics handler can send alerts if any training metrics regress. These alerts will appear in Stackdriver Error Reporting for your project, where you can ack, resolve, or link incidents to bugs. See [here](https://cloud.google.com/error-reporting/docs/notifications) for how to set up notifications for incidents.

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

  # (Optional) Defaults to True. If set to False, this test will not create
  # any alerts for out-of-bounds metrics.
  "alert_for_oob_metrics": "True",

  # (Optional) Defaults to True. If set to False, this test will not create
  # any alerts emails if the test crashes or times out.
  "alert_for_failed_jobs": "True"
}
```
## Config "Recipes"

See the `Config` section above for full details. Here are some common config patterns you might want to use.

#### I want no alerts, no emails, and don't want to write to BigQuery.

Best to turn off the metrics handler entirely. Delete the Cloud Function and the Cloud Scheduler to trigger the Cloud Function (both of which you added in the `Setup` section above).

#### I want to record metrics in BigQuery but I don't alerts/emails.

This will record the final value of all metrics you write to Tensorboard, plus `total_wall_time` and memory usage metrics like `vm_memory_usage_bytes`.

```
  "metric_collection_config": {
    "default_aggregation_strategies": ["final"],
    "write_to_bigquery": true
  },
  "regression_test_config": {
    "alert_for_failed_jobs": false,
    "alert_for_oob_metrics": false
  },
```

#### Record metrics and alert if a test crashes but not if metrics regress.

Record metrics in the same way as the example above.

```
  "metric_collection_config": {
    "default_aggregation_strategies": ["final"],
    "write_to_bigquery": true
  },
  "regression_test_config": {
    "alert_for_oob_metrics": false
  },
```

#### Record metrics and alert if any metric spikes up above the average.

Record metrics in the same way as the examples above.

Alert if any metric value **increases** by a significant amount above the mean.

Wait for 10 datapoints for a metric before enrolling that metric in alerts.

```
  "metric_collection_config": {
    "default_aggregation_strategies": ["final"],
    "write_to_bigquery": true
  },
  "regression_test_config": {
    "metric_success_conditions": {
      "default": {
        "comparison": "less",
          "success_threshold": {
            "stddevs_from_mean": 4.0
          },
        "wait_for_n_points_of_history": 10
      }
    }
  },
```

#### Alert if any metric spikes up or if a scalar constant metric changes at all.

Record metrics in the same way as the examples above.

For the `metric_with_constant_value` metric, alert if the value goes up at all.

This is useful in the case where you don't know what a metric value will be when writing the test, but the value should be the same every day. If you simply used `stddevs_from_mean=0` and `comparison=less`, the alert would always fire since your metric is not less than the mean unless it's decreasing every day.

**If you do know what a metric value should be, then see the next example.**

For all other metrics, alert if the value increases significantly above the mean.

Wait for 10 datapoints for a metric before enrolling that metric in alerts.

```
  "metric_collection_config": {
    "default_aggregation_strategies": ["final"],
    "write_to_bigquery": true
  },
  "regression_test_config": {
    "metric_success_conditions": {
      "default": {
        "comparison": "less",
          "success_threshold": {
            "stddevs_from_mean": 4.0
          },
        "wait_for_n_points_of_history": 10
      },
      "metric_with_constant_value": {
        "comparison": "less_or_equal",
        "success_threshold": {
          "stddevs_from_mean": 0
        }
      }
    }
  },
```

#### Alert if a metric falls below or above a known value.

Record metrics in the same way as the examples above.

Alert if `Accuracy/test_final` falls below 99.0 or if `Loss/test_final` rises above 0.5.

Do not alert for any other metric.

These take effect immediately and will not wait for N runs before alerting.

```
  "metric_collection_config": {
    "default_aggregation_strategies": ["final"],
    "write_to_bigquery": true
  },
  "regression_test_config": {
    "metric_success_conditions": {
      "Accuracy/test_final": {
        "comparison": "greater",
        "success_threshold": {
          "fixed_value": 99.0
        }
      },
      "Loss/test_final": {
        "comparison": "less",
        "success_threshold": {
          "fixed_value": 0.5
        }
      }
    }
  },
```
