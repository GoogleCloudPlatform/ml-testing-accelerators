local mnist = import "mnist.libsonnet";
local resnet50 = import "resnet50.libsonnet";

# Add new models here
std.flattenArrays([
  mnist.configs,
  resnet50.configs,
])
