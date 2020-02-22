# Example Kubernetes Jobs

This directory is recommended for users who do not need inheritance in their
tests and/or have a relatively small number of tests (less than ~20).

See the `templates/` directory for a templating engine to help with generating
a large number of tests that can use inheritance.

## Getting started

1. Make a copy of the example that is closest to your needs.
2. Address all of the TODO comments in that YAML.
3. Add your resulting YAML file to create a Kubernetes job with: `kubectl apply
  -f my_test_name.yaml`.
4. Either wait for your test to kick off automatically (based on the schedule you
  set) or navigate to the [Kubernetes Workload](https://console.cloud.google.com/kubernetes/workload) and click "Run Now".
    * NOTE: It's best to run `kubectl apply`, **then wait a few minutes**, then click "Run Now" to avoid race conditions.
