local base = import "../base.libsonnet";

{
  GardenTest:: base.GardenTest {
    frameworkPrefix: "tf-nightly",
    frameworkVersion: "nightly-2.x",
    imageTag: "nightly",
  },
}
