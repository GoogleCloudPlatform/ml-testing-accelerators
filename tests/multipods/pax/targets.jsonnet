local lmspmdNightly = import 'lmspmdNightly.libsonnet';

std.flattenArrays([
  lmspmdNightly.configs,
])
