# XL ML Test

Cloud **Accel**erated **M**achine **L**earning **Test**s

A set of tools and examples to run machine learning tests on ML hardware
accelerators (TPUs or GPUs) using Google Cloud Platform.

This is not an officially supported Google product.

## Getting Started

1. Reach out to xl-ml-test-users@googlegroups.com to see about potentially getting a free trial on Google Cloud Platform.
2. Follow instructions in the `deployments` directory to set up a Kubernetes Cluster.
3. Follow instructions in the `images` directory to set up the Docker image that your tests will run.
4. Set up your tests. Here you have 1 of 2 choices:
    * See `examples` directory if you will have a relatively small number of tests and don't need inheritance in your tests.
    * See `templates` directory for a more powerful templating engine to generate test config files programmatically.
