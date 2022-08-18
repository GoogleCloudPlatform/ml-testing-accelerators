# Test Templates

## Build Instructions

See [our developing doc](../doc/developing) for build pre-requisites.

To build all of the templates and output Kubernetes resources, run the following:

```bash
scripts/gen-tests.sh
```

This command will output Kubernetes `CronJob` resources into [`k8s/`](../k8s) directory.

Note: Googlers and contributors working out of this repository don't need to manually deploy generated Kubernetes resources with `kubectl`, since we have triggers set up to do that automatically.


## Listing All Existing Tests

To list all of the correctly configured tests, you can run

```bash
$ ./scripts/list-tests.sh
+ jsonnet -J . -S tests/list_tests.jsonnet
flax-latest-imagenet-conv-v3-32-1vm
flax-latest-imagenet-conv-v3-8-1vm
flax-latest-imagenet-func-v2-8-1vm
flax-latest-imagenet-func-v3-32-1vm
flax-latest-vit-conv-v3-8-1vm
flax-latest-vit-func-v2-8-1vm
...
```

This can be helpful for checking that your newly added test is configured
correctly, or to extract the correct name to run a one shot test.


## Running a One Shot Test

To manually run one shot tests, first [connect to a cluster](https://console.cloud.google.com/kubernetes/list) and then run the following:

```bash
export TEST_NAME=tf-nightly-mnist-func-v2-8
jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$TEST_NAME | kubectl create -f -
```

The above command will generate a job id such as `job.batch/pt-nightly-unet3d-conv-v3-8-1vm-gz8ww`. To find the detail of the test, search in [GoogleCloud->Kubernetes->workload in the project `xl-ml-test`](https://pantheon.corp.google.com/kubernetes/workload/overview?mods=allow_workbench_image_override&project=xl-ml-test) with the job id `pt-nightly-unet3d-conv-v3-8-1vm-gz8ww`.

For convenience, the steps of connecting to a cluster and running a one shot
test have been combined into a single script as follows:

```bash
export TEST_NAME=tf-nightly-mnist-func-v2-8
./scripts/run-oneshot.sh -t $TEST_NAME
```

Other flags:
- `-d | --dryrun` if set, then the test does not run but only prints commands.
- `-h | --help`   prints the help screen.


## Running Multiple One Shot Tests

In case you want to run multiple tests, you might find it convenient to combine the above scripts as follows:

```bash
./scripts/list-tests.sh | grep "tf" | grep "nightly" | grep "mnist" while read -r test; do ./scripts/run-oneshot.sh -t $test; done
```

Please be mindful of the resources in the project before running this.


## Creating a New Test

To create a new test, start by copying a similar file from the same ML framework and version. Update the training commands as necessary, and add that file to the `targets.jsonnet` in the same directory.

See [here](../metrics_handler/README.md) for details on configuring alerts and recording the training metrics of your test.

Before you send your code for review, we recommend that you run a one-shot test using the command above to ensure that the test works as expected. If you're not sure what the generated name of your test will be, try running `multifile.jsonnet` to see what the file names of the generated tests are.
