# Creating Tests with JSonnet Templates

## Prerequisites

1. [A GCP project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1. A clone of this repository.
1. Build dependencies from [developing.md](/doc/developing.md).
1. A GKE cluster with GPUs and (optionally) Cloud TPUs. Accelerator availability depends on GCE zone. All of the accelerators used in this tutorial are available in `us-central1-b`.
  - Example commands:
    ```bash
    gcloud beta container clusters create tutorial-cluster \
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

Before you begin, set the following environment variables:

```bash
export PROJECT_ID=...
export GCS_BUCKET=gs://... 
```

## Creating a simple test without accelerators

We'll start by creating a simple test that runs the example MNIST model from the TensorFlow model garden on CPUs. First, build the TensorFlow test image from this repository:

```bash
gcloud builds submit --config images/tensorflow/cloudbuild.yaml --project=$PROJECT_ID --substitutions _VERSION=r2.2,_BASE_IMAGE_VERSION=2.2.0,_MODEL_GARDEN_BRANCH=r2.2.0
```

This new image will be available at `gcr.io/your-project/tensorflow:r2.2`. You can view your GCR images at https://console.cloud.google.com/gcr.

Next, create the file `mnist-cpu.jsonnet`:

```jsonnet
local base = import 'templates/base.libsonnet';

local mnist = base.BaseTest {
  frameworkPrefix: "tf",
  modelName: "mnist",
  mode: "example",

  timeout: 3600, # 1 hour, in seconds

  image: 'gcr.io/your-project/tensorflow', # Change 'your-project' to your project id
  imageTag: 'r2.2',

  accelerator: base.cpu,

  command: [
    'python3',
    'official/vision/image_classification/mnist_main.py',
    '--data_dir=/tmp/mnist',
    '--download',
    '--num_gpus=0',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ]
};

mnist.oneshotJob
```

This templated test will build a Kubernetes `Job` resource that will download the MNIST dataset to a temporary folder and run the model for one epoch on CPUs. You can build this file by `jsonnet`: 

```bash
jsonnet -J ml-testing-accelerators/ mnist-cpu.jsonnet
```

The output should look something like this:

```json
{
   "apiVersion": "batch/v1",
   "kind": "Job",
   "metadata": {
      "generateName": "tf-mnist-example-cpu-"
   },
   "spec": {
      "activeDeadlineSeconds": 3600,
      "template": {
         "spec": {
            "containers": [
               {
                  "args": [
                     "python3",
                     "official/vision/image_classification/mnist_main.py",
                     "--data_dir=/tmp/mnist",
                     "--download",
                     "--num_gpu=0",
                     "--train_epochs=1",
                     "--model_dir=$(MODEL_DIR)",
                  ],
                  "env": [
                     {
                        "name": "JOB_NAME",
                        "valueFrom": {
                           "fieldRef": {
                              "fieldPath": "metadata.labels['job-name']"
                           }
                        }
                     },
                     {
                        "name": "MODEL_DIR",
                        "value": "$(OUTPUT_BUCKET)/tf/mnist/example/cpu/$(JOB_NAME)",
                     },
                     [...]
                  ],
                  "envFrom": [
                     {
                        "configMapRef": {
                           "name": "gcs-buckets"
                        }
                     }
                  ],
                  "image": "gcr.io/your-project/tensorflow:r2.2",
                  "name": "train",
                  "resources": {
                     "limits": {
                        "cpu": 2,
                        "memory": "2Gi"
                     }
                  }
               }
            ],
         }
      }
   }
}
```

Some unimportant fields have been removed for brevity. Note how the following fields affect the final output:

- `frameworkPrefix`, `modelName`, and `mode` are used to generate the name of the Kubernetes resource. Their exact values are not important, but they are useful for organizing large numbers of tests. Note that the final, generated `metadata.name` has to be a valid Kubernetes identifier.
- `timeout` corresponds to the `Job`'s `activeDeadlineSeconds`.
- `image` and `imageTag` are used to generate the image used on the main container (named `train` above).
- `accelerator` is the type of accelerator used, and it will affect `resources.limits` for the `train` container. For this example, we use the CPU "accelerator", which requests a small amount of CPU and memory.
- `command` is used to build the command run on the `train` container.

Notice that the `--model_dir` flag has an environment variable substitution for `MODEL_DIR`, which itself has multiple substitutions. `JOB_NAME` comes from the `env` section of the `train` container. `OUTPUT_BUCKET` can be configured at the namespace level by a Kubernetes `ConfigMap` to determine where models write checkpoints and TensorBoard summaries.

To create a `ConfigMap` with to specify `OUTPUT_BUCKET`, run the following command:

```bash
kubectl create configmap gcs-buckets --from-literal=OUTPUT_BUCKET=$GCS_BUCKET/output
```

Finally, you can create and run the job with the following command: 

```
jsonnet -J ml-testing-accelerators/ mnist-cpu.jsonnet | kubectl create -f -
```

You can find the job running on your [GKE workloads](https://console.cloud.google.com/kubernetes/workload) page.

## Storing datasets on GCS

The above example downloads the MNIST dataset every time it runs. That's workable for a small dataset like MNIST, but that's not practical for large datasets. The easiest way to store the dataset is on GCS.

For this example, we can download the MNIST dataset with TensorFlow Datasets (TFDS). Either install the `tensorflow-datasets` pip package or run a Docker image that already has it installed:

```bash
kubectl run -it --rm --image gcr.io/$PROJECT_ID/tensorflow:r2.2 --env=GCS_BUCKET=$GCS_BUCKET $USER bash
```

Then, download and prepare the dataset with TFDS:

```bash
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=mnist --data_dir=$GCS_BUCKET/tfds
```

Optionally, confirm that MNIST has been downloaded by running `gsutil`:

```bash
gsutil ls $GCS_BUCKET/tfds
```

Update the `gcs-buckets` `ConfigMap` with a variable to point to the new directory:

```bash
kubectl delete configmap gcs-buckets
kubectl create configmap gcs-buckets --from-literal=OUTPUT_BUCKET=$GCS_BUCKET/output --from-literal=MNIST_DIR=$GCS_BUCKET/tfds
```

Then, update the `command` field in `mnist-cpu.jsonnet`;

```jsonnet
  command: [
    'python3',
    'official/vision/image_classification/mnist_main.py',
    '--data_dir=$(MNIST_DIR)',
    '--num_gpus=0',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],
```

Run `jsonnet` and `kubectl` again to run the test with `$MNIST_DIR`.

```
jsonnet -J ml-testing-accelerators/ mnist-cpu.jsonnet | kubectl create -f -
```

## Updating the test to use GPUs

Using different accelerators is easy with our templates. Copy `mnist-cpu.jsonnet` to a new file called `mnist-gpu.jsonnet` and add or update the following lines:

```jsonnet
local base = import 'templates/base.libsonnet';
local gpus = import 'templates/gpus.libsonnet';

local mnist = base.BaseTest {
  [...]
  accelerator: gpus.teslaV100,

  command: [
    'python3',
    'official/vision/image_classification/mnist_main.py',
    '--data_dir=$(MNIST_DIR)',
    '--num_gpus=1',
    '--train_epochs=1',
    '--model_dir=$(MODEL_DIR)',
  ],
  [...]
```

Note how we changed the `--num_gpu` flag to 1 and replaced the `accelerator` field with  `gpus.teslaV100`. Run `jsonnet` to inspect the update Kubernetes `Job`: 

```
jsonnet -J ml-testing-accelerators/ mnist-gpu.jsonnet
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
jsonnet -J ml-testing-accelerators/ mnist-gpu.jsonnet | kubectl create -f -
```

## Updating the test to use TPUs

Again, make a copy of `mnist-cpu.jsonnet` and name it to `mnist-tpu.jsonnet`. Add or update the following lines:

```jsonnet
local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

local mnist = base.BaseTest {
  [...]
  tpuSettings+: {
    softwareVersion: '2.2',
  },
  accelerator: tpus.v2_8,

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

We added `tpu_version: '2.2'` and `accelerator: tpus.v2_8` to the test config and updated the model `command`, removing `--num_gpu` and adding `--distribution_strategy=tpu`. Build the Kubernetes `Job` with `jsonnet`: 

```
jsonnet -J ml-testing-accelerators/ mnist-tpu.jsonnet
```

Note the following lines in the generated `Job`:

```
   [...]
   "template":
      "metadata":
         "annotations":
            "tf-version.cloud-tpus.google.com": "2.2"
   [...]
      "resources":
         "limits":
            "cloud-tpus.google.com/v2": 8
```

The TPU software version is specified in `tf-version.cloud-tpus.google.com`. `cloud-tpus.google.com/v2: 8` requests 8 v2 TPU cores, i.e. a v2-8 TPU device.

If you have sufficient TPU quota, try running the test and observing the output:

```bash
jsonnet -J ml-testing-accelerators/ mnist-tpu.jsonnet | kubectl create -f -
```

## Next steps

- Delete any resources you created just to follow this tutorial.
- Learn how to automate tests and monitor regressions in the [next tutorial](test-templates-automated).
- Explore our other documents in [docs/](/docs/).
