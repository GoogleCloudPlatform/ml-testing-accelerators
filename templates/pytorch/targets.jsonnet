local nightly = import "nightly/targets.jsonnet";

// Add new versions here
std.flattenArrays([
  nightly,
])
