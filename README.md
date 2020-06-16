# ML Testing Accelerators

A set of tools and examples to run machine learning tests on ML hardware
accelerators (TPUs or GPUs) using Google Cloud Platform.

This is not an officially supported Google product.

## Getting Started

1. Install all of our [development prerequisites](doc/developing.md#Prerequisites).
1. Follow instructions in the [`deployments`](deployments/README.md) directory to set up a Kubernetes Cluster.
1. Follow instructions in the [`images`](images/README.md) directory to set up the Docker image that your tests will run.
1. Deploy the [metrics handler](metrics_handler/README.md) to [Google Cloud Functions](https://cloud.google.com/functions).
1. Set up your tests. Here you have 1 of 2 choices:
    * See [`examples`](examples/README.md) directory if you will have a relatively small number of tests and don't need inheritance in your tests.
    * See [`tests`](tests/README.md) directory for a more powerful templating engine to generate test config files programmatically.

Are you interested in using ML Testing Accelerators? E-mail [ml-testing-accelerators-users@googlegroups.com](mailto:ml-testing-accelerators-users@googlegroups.com) and tell us about your use-case. We're happy to help you get started.
