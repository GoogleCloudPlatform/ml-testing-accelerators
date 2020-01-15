local mnist = import 'garden/nightly/mnist.libsonnet';

# Times in UTC
local schedules = {
  functional: '0 */12 * * *',
  convergence: '30 7 * * */1',
};

{
  [config.job_name + '.yaml']: std.manifestYamlDoc(config.cron_job(schedules[config.mode]))
    # Add model configs here
    for config in std.flattenArrays([
      mnist.configs,
    ])
}
