local syncDevicesTests = import 'sync-devices.libsonnet';

std.flattenArrays([
  syncDevicesTests.configs,
])
