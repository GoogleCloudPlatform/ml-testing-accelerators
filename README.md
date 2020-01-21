# XL ML Test

Cloud **Accel**erated **M**achine **L**earning Tests

To run one test once, run the following:

```bash
jsonnet templates/oneshot.jsonnet --tla-str test=$TEST_NAME | kubectl apply -f -
```

To generate and deploy Kubernetes YAMLs from the templates, run the following:

```bash
jsonnet -S templates/multifile.jsonnet -m k8s/gen
kubectl apply -f k8s/gen
```
