# Cloud Deployment Manager templates

Deployment Manager docs: https://cloud.google.com/deployment-manager/docs/

To create testing cluster, run one of the following commands:

```bash
# Minimal cluster, compatible with most TensorFlow TPU models
gcloud deployment-manager deployments create my-example-cluster --template deployments/europe-west4/cluster.jinja
# Larger cluster, with multi-GPU machines and large GCE VMs for PyTorch TPU models
gcloud deployment-manager deployments create testing-cluster --template deployments/us-central1/cluster.jinja
gcloud deployment-manager deployments create testing-cluster-pools --template deployments/us-central1/node-pools.jinja --properties=cluster-name:testing-cluster,small-tpu:true,huge-tpu:false,gpu-v100x4:false,gpu-k80x8:false
```

If you're using GPUs, connect `kubectl` to your cluster, then run the following command to install device drivers:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

You should only need to run this command once when you create the cluster.
