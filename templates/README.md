# Test Templates

## Build Instructions

See [our developing doc](../doc/developing) for build pre-requisites.

To run a single test, run the following:

```bash
jsonnet templates/oneshot.jsonnet -S --tla-str test=$TEST_NAME | kubectl create -f -
```

`$TEST_NAME` is the generated name of a test, such as `tf-nightly-mnist-functional-v2-8`.

To build all of the templates and output Kubernetes resources, run the following:

```bash
jsonnet templates/multifile.jsonnet -S -m k8s/gen
```

This command will output Kubernetes `CronJob` resources into [`k8s/gen`](../k8s/gen) directory. You can deploy these resources by running the following command:

```bash
kubectl apply -f k8s/gen -f k8s/
```

Note: Googlers and contributors working out of this repository don't need to manually deploy generated Kubernetes resources with `kubectl`, since we have triggers set up to do that automatically.

## Creating a New Test

To create a new test, start by copying a similar file from the same ML framework and version. Update the training commands as necessary, and add that file to the `targets.jsonnet` in the same directory.

Before you send your code for review, we recommend that you run a one-shot test using the command above to ensure that the test works as expected. If you're not sure what the generated name of your test will be, try running `multifile.jsonnet` to see what the file names of the generated tests are.
