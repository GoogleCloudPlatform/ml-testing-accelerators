local base = import "../base.libsonnet";

{
  PyTorchTest:: base.PyTorchTest {
    frameworkPrefix: "pt-nightly",
    frameworkVersion: "pytorch-nightly",
    imageTag: "nightly", 
  },
}