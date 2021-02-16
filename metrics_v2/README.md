# Metrics V2

This directory contains a WIP rewrite of the metrics handler. Contents are experimental and may not work as expected. For the current version of the metrics handler, see [metrics_handler/](../metrics_handler).

## Setup

These packages require Python >= 3.8. Your local pip packages may conflict with those downloaded by Bazel as part of the build process. It's strongly recommended to create a new virtual environment with minimal dependencies. For example, using `conda`, run `conda create -n py38_clean python=3.8`.

The instructions in this document may require the following environment variables to be set:

```bash
PROJECT=...  # GCP project ID. Optional if you are using application default credentials.
PUBSUB_TOPIC=...  # The name of the PubSub topic to write messages to.
GCS_BUCKET=gs://...  # The GCS bucket that your models write to.
CLUSTER_NAME=...  # The name of the cluster where your models are running.
CLUSTER_LOCATION=...  # The location (GCP zone or region) of $CLUSTER_NAME.
```
## Run tests

To run all tests in this repository, run `bazel test //...`.

## Event Publisher

To run the event publisher locally, first connect to your GKE cluster with the `gcloud container cluster get-credentials`.
Then, run the following:

```bash
bazel run //publisher -- --project=$PROJECT --pubsub_topic=$PUBSUB_TOPIC --model_output_bucket=$GCS_BUCKET --cluster_name=$CLUSTER_NAME --cluster_location=$CLUSTER_LOCATION
```

To build and push the Publisher image, run the following:

```bash
bazel run --define project=$PROJECT publisher:image_push
```

TODO: Write instructions for deploying to GKE.

## Metrics Handler

TODO: Local testing instructions

To build and deploy the handler Cloud Function, run the following:

```bash
bazel run handler:deploy --name metrics-handler --topic test-completed --dataset metrics --project $(gcloud config get-value project)
```
