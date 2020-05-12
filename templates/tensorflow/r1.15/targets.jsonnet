local efficientnet = import "efficientnet.libsonnet";
local mask_rcnn = import "mask-rcnn.libsonnet";
local mnasnet = import "mnasnet.libsonnet";
local resnet = import "resnet.libsonnet";

std.flattenArrays([
  efficientnet.configs,
  resnet.configs,
  mask_rcnn.configs,
  mnasnet.configs,
])
