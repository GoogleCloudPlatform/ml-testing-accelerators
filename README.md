# ML Testing Accelerators

A set of tools and examples to run machine learning tests on ML hardware
accelerators (TPUs or GPUs) using Google Cloud Platform.

not yet! This is not an officially supported Google product.

## Getting Started

In this mode, your tests and/or models run on an automated schedule in GKE.
Results are collected by the "Metrics Handler" and written to BigQuery.

1. Install all of our [development prerequisites](doc/developing.md#Prerequisites).
1. Follow instructions in the [`deployments`](deployments/README.md) directory to set up a Kubernetes Cluster.
1. Follow instructions in the [`images`](images/README.md) directory to set up the Docker image that your tests will run.
1. Deploy the [metrics handler](metrics/README.md#metrics-handler) to [Google Cloud Functions](https://cloud.google.com/functions).
1. Deploy the [event publisher](metrics/README.md#event-publisher) to you GKE cluster.
1. See [`templates`](templates/README.md) directory for a [JSonnet](https://jsonnet.org/) template library to generate test config files.
1. (Optional) Set up a dashboard to view test results. See [ dashboard ](dashboard/README.md) directory for instructions.

Are you interested in using ML Testing Accelerators? E-mail [ml-testing-accelerators-users@googlegroups.com](mailto:ml-testing-accelerators-users@googlegroups.com) and tell us about your use-case. We're happy to help you get started.
