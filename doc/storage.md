# Storage Options

## Google Cloud Storage

[Google Cloud Storage](https://cloud.google.com/storage/docs) is a convenient option for models using TensorFlow, since it has native support, and it is the best storage option for use with TensorFlow TPUs. For best performance, we recommend that you use a regional bucket in the same region in which you are running your model. For example, if your cluster is located in `us-central1-b`, you should create a regional GCS bucket in `us-central1`. For models using frameworks that do not have native GCS support (e.g. PyTorch), you should add a step to your test to copy the data from GCS to local storage before running the model. You do not need any additional setup to use GCS with GKE, as long as your compute service account (or the service accound of your Kubernetes nodes) and the TPU service account (if applicable) have access to your data bucket.

## Cloud Filestore

The easiest option to emulate local storage with multiple readers in Kubernetes is [Cloud Filestore](https://cloud.google.com/filestore/docs). To get started, [create a Filestore instance](https://cloud.google.com/filestore/docs/creating-instances) in the same zone as your cluster. Then, [mount the fileshare to a GCE instance](https://cloud.google.com/filestore/docs/mounting-fileshares) and [copy your dataset over](https://cloud.google.com/filestore/docs/copying-data).

In your GKE cluster, create a `PersistentVolume` and `PersistentVolumeClaim` like the following:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: filestore-datasets
spec:
  capacity:
    storage: 1T # or the size of your filestore instance
  accessModes:
  - ReadOnlyMany
  nfs:
    #
    path: /FILESHARE_NAME
    server: FILESTORE_IP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: filestore-datasets-claim
spec:
  accessModes:
  - ReadOnlyMany
  storageClassName: ""
  volumeName: filestore-datasets
  resources:
    requests:
      storage: 1Ki
```

`Job`s that rely on the dataset in Filestore will need to have the following line in their pod template to bind to that `PersistentVolumeClaim`:

```yaml
  ...
  - name: filestore-datasets
    persistentVolumeClaim:
      claimName: filestore-datasets-claim
  ...
```

If you are using our templates to generate your automated tests, you can add a [`PersistentVolumeSpec`](../tests/volumes.libsonnet) to a test's `volumeMap` to automatically mount the volume in the test's main `train` container.
