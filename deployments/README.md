# Cloud Deployment Manager templates

Deployment Manager docs: https://cloud.google.com/deployment-manager/docs/

To create testing cluster, run one of the following commands:

```bash
# Minimal cluster, compatible with most TensorFlow TPU models
gcloud deployment-manager deployments create my-example-cluster --config deployments/minimal.yaml
# Larger cluster, with multi-GPU machines and large GCE VMs for PyTorch TPU models
gcloud deployment-manager deployments create my-big-example-cluster --template deployments/us-central1-cluster.jinja
```

If you're using GPUs, connect to your cluster, then run the following command to install device drivers:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

You should only need to run this command once when you create the cluster.
