local imagenet = import "imagenet.libsonnet";
local mnist = import "mnist.libsonnet";

# Add new libsonnet files here.
std.flattenArrays([
  imagenet.configs,
  mnist.configs,
])
