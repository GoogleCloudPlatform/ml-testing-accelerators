# Dashboard

This is an example dashboard to display the data that is collected by the [metrics handler](../metrics_handler).

This guide assumes you have a Google Cloud Platform project and you are running the metrics handler to collect data in BigQuery.

## Running locally

Command to run locally: `python3 -m bokeh serve --show dashboard/dashboard.py dashboard/metrics.py`


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
