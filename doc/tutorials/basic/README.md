
# Creating Tests with JSonnet Templates

## Prerequisites

1. [A GCP project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1. A copy of this repository.
1. Dependencies from [developing.md](/doc/developing.md).
1. A GKE cluster with GPUs and (optionally) Cloud TPUs. Accelerator availability depends on GCE zone. All of the accelerators used in this tutorial are available in `us-central1-b`.
  - Example commands:
    ```bash
    gcloud container clusters create tutorial-cluster \
      --zone us-central1-b \
      --release-channel regular \
      --machine-type n1-standard-4 \
      --accelerator "type=nvidia-tesla-v100,count=1" \
      --scopes "https://www.googleapis.com/auth/cloud-platform" \
      --num-nodes 1 \
      --enable-ip-alias \
      --enable-autoupgrade \
      --enable-tpu \
      --project=$PROJECT_ID
    ```
    ```bash
    # Connect to your cluster
    gcloud container clusters get-credentials tutorial-cluster --project $PROJECT_ID --zone us-central1-b
    ```
    ```bash
    # Install GPU drivers
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    ```
1. A GCS bucket.
  - To create a new GCS bucket, run `gsutil mb -c standard -l us-central1 gs://your-bucket-name`

`ml-testing-accelerators` relies heavily on GKE to run workloads, so it is also expected that you should have basic familiarity with creating and running Kubernetes batch workloads. If you need a refresher, see the following documents:

- [Learn Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) (basic unit of a workload)
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/) (batch workloads)

`ml-testing-accelerators` also uses [JSonnet](https://jsonnet.org) templates to build Kubernetes workloads. It may be helpful to work through [JSonnet's tutorials](https://jsonnet.org/learning/tutorial.html) first, but the examples in this document are simple enough that it is not required.

Before you begin, set the following environment variables:

```bash
$ export PROJECT_ID=...
$ export GCS_BUCKET=gs://...
```

## Creating a simple test without accelerators

We'll start by creating a simple test that runs the example MNIST model from the TensorFlow Model garden on CPU. See the file `mnist-cpu.jsonnet`:

```jsonnet
local base = import 'templates/base.libsonnet';

local mnist = base.BaseTest {
  // Configure job name
  frameworkPrefix: "tf",
  modelName: "mnist",
  mode: "example",
  timeout: 3600, # 1 hour, in seconds

  // Set up runtime environment
  image: 'tensorflow/tensorflow', // Official TF docker image
  imageTag: '2.9.1',
  accelerator: base.cpu,
  outputBucket: std.extVar('gcs-bucket'),

  // Override entrypoint to install TF official models before running `command`.
  entrypoint: [
    'bash',
    '-c',
    |||
      pip install tf-models-official==2.9.1

      # Run whatever is in `command` here
      ${@:0}
    |||
  ],
  command: [
    'python3',
    '-m',
    'official.legacy.image_classification.mnist_main',
    '--data_dir=/tmp/mnist',
    '--download',
    '--num_gpus=0',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],
};

std.manifestYamlDoc(mnist.oneshotJob, quote_keys=false)
```

This templated test will build a Kubernetes `Job` resource that will run the model on the [official TensorFlow Docker image](https://hub.docker.com/r/tensorflow/tensorflow/) on CPUs. The input data will be downloaded locally as needed, and the output TensorBoard metrics and checkpoint will be written to your Google Cloud Storage bucket. Those metrics will be used in the [next tutorial](../automated). We will take a closer look at how each template field affects the output once we build it. From the root of this repository, you can build this file with `jsonnet`:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/basic/mnist-cpu.jsonnet
```

Let's break down the flags in this command:

-- `--jpath=.` searches for dependencies in this repository
-- `--string` outputs a string (i.e. a YAML document from `std.manifestYamlDoc`) instead of a JSON object
-- `--ext-str gcs-bucket=$GCS_BUCKET` substitutes your GCS bucket as an [ExtVar](https://jsonnet.org/ref/language.html) into `outputBucket`

The output should look something like this:

```yaml
apiVersion: "batch/v1"
kind: "Job"
metadata:
  generateName: "tf-mnist-example-cpu-"
  labels:
    accelerator: "cpu"
    [ ... ]
    model: "mnist"
spec:
  activeDeadlineSeconds: 3600
  [ ... ]
  template:
    spec:
      containers:
      - name: "train"
        command:
        - "bash"
        - "-c"
        - |
          pip install tf-models-official==2.9.1

          # Run whatever is in `command` here
          $*
        args:
        - "python3"
        - "official/vision/image_classification/mnist_main.py"
        - "--data_dir=/tmp/mnist"
        - "--download"
        - "--num_gpus=0"
        - "--train_epochs=1"
        - "--model_dir=$(MODEL_DIR)"
        env:
        [ ... ]
        - name: "JOB_NAME"
          valueFrom:
            fieldRef:
              fieldPath: "metadata.labels['job-name']"
        - name: "MODEL_DIR"
          value: "gs://wcromar-tmp/tutorial2022/tf/mnist/example/cpu/$(JOB_NAME)"
        [ ... ]
        image: "tensorflow/tensorflow:r2.9.1"
        resources:
          limits:
            cpu: 2
            memory: "2Gi"
```

Some fields have been removed or re-ordered for clarity. Note how the following fields affect the final output:

- `frameworkPrefix`, `modelName`, and `mode` are used to generate the name of the Kubernetes resource. Their exact values are not important, but they are useful for organizing large numbers of tests. Note that the final, generated `metadata.generateName` has to be a valid Kubernetes identifier.
- `command` is used to build the command run on the `train` container.
- `image` and `imageTag` are used to generate the image used on the main container (named `train` above).
- `accelerator` is the type of accelerator used, and it will affect `resources.limits` for the `train` container. For this example, we use the CPU "accelerator", which requests a small amount of CPU and memory.
- `timeout` corresponds to the `Job`'s `activeDeadlineSeconds`.
- `outputBucket` is used to specify the GCS location to store the output.

Notice that the `--model_dir` flag has an environment variable substitution for `MODEL_DIR`, which itself has a substitution. `JOB_NAME` is copied from the metadata of the generated Pod. Both the `MODEL_DIR` and `OUTPUT_DIR` env vars are automatically added to the `train` container by the templates.

Finally, you can create and run the job with the following command:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/basic/mnist-cpu.jsonnet | kubectl create -f -
```

You should see output like `job.batch/tf-mnist-example-cpu-cm59d created`. Use `kubectl` to follow the job logs, replacing the example name with your own:

```bash
$ kubectl logs job.batch/tf-mnist-example-cpu-cm59d -f
```

You should see a line like this at the end of the logs when training completes:

```
I0621 20:57:59.030551 140608661440320 mnist_main.py:170] Run stats:
{'accuracy_top_1': 0.8236762285232544, 'eval_loss': 0.6072796583175659, 'loss': 1.7133259773254395, 'training_accuracy_top_1': 0.48530104756355286}
```

You also can find the job running on your [GKE workloads](https://console.cloud.google.com/kubernetes/workload) page in the Cloud Console.

## Updating the test to use GPUs

Using different accelerators is easy with our templates. See the changes in `mnist-gpu.jsonnet` to use GPUs:

```jsonnet
local base = import 'templates/base.libsonnet';
local gpus = import 'templates/gpus.libsonnet';

local mnist = base.BaseTest {
  [...]
  accelerator: gpus.teslaV100,
  [...]
  command: [
    'python3',
    'official/vision/image_classification/mnist_main.py',
    '--data_dir=/tmp/mnist',
    '--num_gpus=1',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],
  [...]
```

Note how we changed the `--num_gpu` flag to 1 and replaced the `accelerator` field with  `gpus.teslaV100`. Run `jsonnet` to inspect the update Kubernetes `Job`:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/basic/mnist-gpu.jsonnet
```

Notice the following lines in the output:

```
   [...]
      "resources":
         "limits":
         "nvidia.com/gpu": 1
   [...]
   "nodeSelector":
     "cloud.google.com/gke-accelerator": "nvidia-tesla-v100"
```

This will run the test on a node with at least 1 Nvidia Tesla V100 GPU available. Run the `Job` with `jsonnet` and `kubectl`

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/basic/mnist-gpu.jsonnet | kubectl create -f -
```

Use `kubectl` or Cloud Console to check the job results.

## Updating the test to use TPUs (optional)

Similarly, we can configure the test to use TPUs too. See the following changes in `mnist-tpu.jsonnet`:

```jsonnet
local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

local mnist = base.BaseTest {
  [...]
  tpuSettings+: {
    softwareVersion: '2.9.1',
  },
  accelerator: tpus.v2_8,
  [...]
  command: [
    'python3',
    'official/vision/image_classification/mnist_main.py',
    '--distribution_strategy=tpu',
    '--data_dir=$(MNIST_DIR)',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],
  [...]
}
```

We added `tpu_version: '2.9.1'` and `accelerator: tpus.v2_8` to the test config and updated the model `command`, removing `--num_gpu` and adding `--distribution_strategy=tpu`. Also note that the input data location (`--data_dir` in this example) [must be in GCS for TPU Nodes](https://cloud.google.com/tpu/docs/storage-options#gcsbuckets). Build the Kubernetes `Job` with `jsonnet`:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/basic/mnist-tpu.jsonnet
```

Note the following lines in the generated `Job`:

```
   [...]
   "template":
      "metadata":
         "annotations":
            "tf-version.cloud-tpus.google.com": "2.9.1"
   [...]
      "resources":
         "limits":
            "cloud-tpus.google.com/v2": 8
```

The TPU software version is specified in `tf-version.cloud-tpus.google.com`. `cloud-tpus.google.com/v2: 8` requests 8 v2 TPU cores, i.e. a v2-8 TPU Node.

If you have sufficient TPU quota, try running the test and observing the output:

```bash
$ jsonnet --jpath . --string --ext-str gcs-bucket=$GCS_BUCKET doc/tutorials/basic/mnist-tpu.jsonnet | kubectl create -f -
```

## Next steps

- Build your own Docker container to test your own code. Learn how to [build container images with Cloud Build](https://cloud.google.com/build/docs/building/build-containers).
- The above example downloads the MNIST dataset every time it runs. That's workable for a small dataset like MNIST, but that's not practical for large datasets. Learn more about storage options in [`../storage.md`](../storage.md).
- Try triggering test jobs from your project's CI pipeline.
- Learn how to automate tests and monitor regressions in the [next tutorial](../automated).
- Explore our other documents in [docs/](/docs/).
