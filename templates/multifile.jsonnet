local targets = import "all_targets.jsonnet";

# Outputs {filename: yaml_string} for each target
{
  [name + ".yaml"]: std.manifestYamlDoc(
    targets[name].cronJob
  ) for name in std.objectFields(targets)
}
