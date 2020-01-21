local mnist = import 'mnist.libsonnet';
local resnet_ctl = import 'resnet-ctl.libsonnet';
local resnet_cfit = import 'resnet-cfit.libsonnet';

# Add new models here
std.flattenArrays([
  mnist.configs,
  resnet_ctl.configs,
  resnet_cfit.configs,
])
