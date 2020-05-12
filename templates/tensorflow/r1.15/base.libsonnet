local base = import "../base.libsonnet";
local mixins = import "../../mixins.libsonnet";

{
  LegacyTpuTest:: base.LegacyTpuTest {
    frameworkPrefix: "tf-r1.15",
    tpuVersion: "1.15",
    imageTag: "1.15",
  },
  Convergence:: mixins.Convergence {
    # Run at 1:00 PST on Saturday
    schedule: "0 8 * * 6"
  },
}