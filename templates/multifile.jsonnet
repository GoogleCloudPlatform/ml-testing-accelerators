local targets = import 'targets.jsonnet';

# Times in UTC
local schedules = {
  functional: '0 */12 * * *',
  convergence: '30 7 * * */1',
};

# Outputs {filename: yaml_string} for each target
{
  [name + '.yaml']: std.manifestYamlDoc(
    targets[name].cron_job(
      schedules[targets[name].mode]
    )
  ) for name in std.objectFields(targets)
}
