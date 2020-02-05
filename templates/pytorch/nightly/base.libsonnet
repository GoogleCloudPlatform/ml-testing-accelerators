local base = import "../base.libsonnet";

{
  PyTorchTest:: base.PyTorchTest {
    frameworkPrefix: "pt-nightly",
    tpuVersion: "pytorch-nightly",
    imageTag: "nightly",
  },
}