# Dashboard

This is an example dashboard to display the data that is collected by the [metrics handler](../metrics_handler).

This guide assumes you have a Google Cloud Platform project and you are running the metrics handler to collect data in BigQuery.

## Running locally

Command to run locally: `python3 -m bokeh serve --show dashboard/dashboard.py dashboard/metrics.py dashboard/compare.py`


## Hosting your dashboard

You can host your dashboard using App Engine.

Query caching with redis is strongly recommended but not strictly required - you can ignore the redis steps and the dashboard will work but will log warnings about failing to connect to the cache.

1. Set up env vars
```
export REGION=us-west2
export PROJECT_ID=my-project
export INSTANCE_NAME=my-redis-instance
```

2. `gcloud redis instances create $INSTANCE_NAME --size=2 --region=$REGION --redis-version=redis_4_0 --project=$PROJECT_ID`

3. Edit `app.yaml` in 3 ways:
  * Update redis info if using redis:
    * REDISHOST = Find this value using: `echo $(gcloud redis instances describe $INSTANCE_NAME --region=$REGION --project=$PROJECT_ID --format='value(host)')`
    * REDISPORT = Find this value using: `echo $(gcloud redis instances describe $INSTANCE_NAME --region=$REGION --project=$PROJECT_ID --format='value(port)')`
  * Update `JOB_HISTORY_TABLE_NAME` and `METRIC_HISTORY_TABLE_NAME`.
    * You can find these table names [here](https://console.cloud.google.com/bigquery) by clicking your project name in the left sidebar.
  * Change `--allow-websocket-origin` arg in `entrypoint` to be the URL of your app engine project. You can find this URL in your [App Engine Dashboard](https://console.cloud.google.com/appengine) (top right of that UI). If you don’t see the URL there, try deploying first (see next step) and at that point you should receive your URL.

4. Make sure you are in the dir where `app.yaml` lives and then run `gcloud app deploy`


## Clean up your hosted dashboard

1. `gcloud redis instances delete $INSTANCE_NAME --region=$REGION --project=$PROJECT_ID`

2. Click "Disable Application" on [this page](https://console.cloud.google.com/appengine/settings)

# Advanced Usage

## Multiple dashboard versions

When you run `gcloud app deploy`, the command defaults to using `app.yaml`. You can specify a different yaml with e.g. `gcloud app deploy custom.yaml` or deploy multiple versions with `gcloud app deploy custom.yaml custom2.yaml`.

Check out `pytorch-dashboard.yaml` for an example of a secondary dashboard and note the `allow-websocket-origin` argument for an example of how the URL looks for the secondary dashboard(s).

## Privacy / authentication

You can restrict which tests are shown in the dashboard by using the `TEST_NAME_PREFIXES` environment variable in the .yaml file.

The hosted dashboard runs as an App Engine app. You can restrict the URL to specific group(s) of people by following the instructions [here](https://cloud.google.com/iap/docs/app-engine-quickstart). You can configure the IAP (identity-aware proxy) rules such that each dashboard version has its own group of restricted viewers. See also the `Multiple dashboard versions` section above.

## Pre-generated URLs for the compare dashboard

The `compare.py` dashboard allows you to specify a list of test names and a list of metric names and renders 1 graph+table for each combination of test and metric.

Since it's tedious to type the same list of test and metric names for a commonly-used query, you can generate a URL with the test and metric names encoded in it.

Note that you can use the SQL wildcard `%` to use prefix or suffix matching.

The URL should be of the form `$APP_URL/compare?test_names=$B64_TEST_NAMES&metric_names=$B64_METRIC_NAMES`. For example, if your `APP_URL` is 'https://xl-ml-test.appspot.com', then a simple snippet to generate the URL could be:

```
python3
 > import base64
 > my_url = 'https://xl-ml-test.appspot.com'
 > b64_test_names = base64.b64encode('tf-nightly-%,my-other-test1,my-other-test-2'.encode('utf-8')).decode()
 > b64_metric_names = base64.b64encode('%examples/sec%,accuracy_final'.encode('utf-8')).decode()
 > print(f'{my_url}/compare?test_names={b64_test_names}&metric_names={b64_metric_names}')
```
(note the syntax would be a little different in python2)
