local mnist = import 'mnist.libsonnet';

# Add new models here
std.flattenArrays([
  mnist.configs,
])
