# Cloud Deployment Manager templates

Deployment Manager docs: https://cloud.google.com/deployment-manager/docs/

To create testing cluster, run one of the following commands:

```bash
# Minimal cluster, compatible with most TensorFlow TPU models
gcloud deployment-manager deployments create xl-ml-oneshots --template deployments/cluster.jinja
# Larger cluster, with multi-GPU machines and large GCE VMs for PyTorch TPU models
gcloud deployment-manager deployments create xl-ml-oneshots --template deployments/cluster.jinja --properties zone:us-central1-b,huge-pool:true,double-v100s:true
```

If you're using GPUs, connect to your cluster, then run the following to install device drivers:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

You should only need to run this command once when you create the cluster.
