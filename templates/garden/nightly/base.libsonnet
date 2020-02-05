local base = import "../base.libsonnet";

{
  GardenTest:: base.GardenTest {
    frameworkPrefix: "tf-nightly",
    tpuVersion: "nightly-2.x",
    imageTag: "nightly",
  },
}
