# Cloud Deployment Manager templates

Deployment Manager docs: https://cloud.google.com/deployment-manager/docs/

To create your testing cluster, run the following:

```bash
gcloud deployment-manager deployments create testing-cluster --config deployments/cluster.yaml
```

If you're using GPUs, connect to your cluster, then run the following to install device drivers:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

You should only need to run this command once when you create the cluster.
