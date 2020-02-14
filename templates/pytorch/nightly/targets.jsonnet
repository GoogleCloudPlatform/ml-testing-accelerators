local mnist = import "mnist.libsonnet";
local resnet50 = import "resnet50.libsonnet";
local cifarTorchvision = import "cifar-torchvision.libsonnet";
local cifarInline = import "cifar-inline.libsonnet";

# Add new models here
std.flattenArrays([
  mnist.configs,
  resnet50.configs,
  cifarTorchvision.configs,
  cifarInline.configs,
])
