# Test Templates

## Build Instructions

See [our developing doc](../doc/developing) for build pre-requisites.

To run a single test, first [connect to a cluster](https://console.cloud.google.com/kubernetes/list) and then run the following:

```bash
scripts/run-oneshot.sh --accelerator v2-8 --file tests/tensorflow/nightly/mnist.libsonnet --type functional
```

To build all of the templates and output Kubernetes resources, run the following:

```bash
scripts/gen-tests.sh
```

This command will output Kubernetes `CronJob` resources into [`k8s/`](../k8s) directory.

Note: Googlers and contributors working out of this repository don't need to manually deploy generated Kubernetes resources with `kubectl`, since we have triggers set up to do that automatically.

## Creating a New Test

To create a new test, start by copying a similar file from the same ML framework and version. Update the training commands as necessary, and add that file to the `targets.jsonnet` in the same directory.

See [here](../metrics_handler/README.md) for details on configuring alerts and recording the training metrics of your test.

Before you send your code for review, we recommend that you run a one-shot test using the command above to ensure that the test works as expected. If you're not sure what the generated name of your test will be, try running `multifile.jsonnet` to see what the file names of the generated tests are.
