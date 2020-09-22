# ML Testing Accelerators

A set of tools and examples to run machine learning tests on ML hardware
accelerators (TPUs or GPUs) using Google Cloud Platform.

This is not an officially supported Google product.

## Getting Started (full-featured standalone mode)

In this mode, your tests and/or models run on an automated schedule in GKE. Results
are collected by the "Metrics Handler" and written to BigQuery.

This route is recommended if you have many tests that run for a long time and
produce many metrics that you want to monitor for regressions.

1. Install all of our [development prerequisites](doc/developing.md#Prerequisites).
1. Follow instructions in the [`deployments`](deployments/README.md) directory to set up a Kubernetes Cluster.
1. Follow instructions in the [`images`](images/README.md) directory to set up the Docker image that your tests will run.
1. Deploy the [metrics handler](metrics_handler/README.md) to [Google Cloud Functions](https://cloud.google.com/functions).
1. Set up your tests. Here you have 1 of 2 choices:
    * See [`examples`](examples/README.md) directory if you will have a relatively small number of tests and don't need inheritance in your tests.
    * See [`templates`](templates/README.md) directory for a more powerful templating engine to generate test config files programmatically.
1. (Optional) Set up a dashboard to view test results. See [ dashboard ](dashboard/README.md) directory for instructions.

## Getting Started (lighter-weight Continuous Integration mode)

In this mode, your tests run on GKE but are tied to a CI platform like Github
Actions or CircleCI. Tests can run as presubmits for pending PRs, as postsubmit
checks on submitted PRs, or on a timed schedule.

This route is recommended if you want some tie-in with Github and your tests are
relatively short-running.

1. Install all of our [development prerequisites](doc/developing.md#Prerequisites).
1. Follow instructions in the [`deployments`](deployments/README.md) directory to set up a Kubernetes Cluster.
1. See the [ci_pytorch](ci_pytorch/README.md) directory for the last few setup steps.


Are you interested in using ML Testing Accelerators? E-mail [ml-testing-accelerators-users@googlegroups.com](mailto:ml-testing-accelerators-users@googlegroups.com) and tell us about your use-case. We're happy to help you get started.
