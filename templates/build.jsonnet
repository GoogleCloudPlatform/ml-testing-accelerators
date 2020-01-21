local tf_mnist = import 'garden/nightly/mnist.libsonnet';
local tf_resnet_ctl = import 'garden/nightly/resnet-cfit.libsonnet';
local tf_resnet_cfit = import 'garden/nightly/resnet-ctl.libsonnet';
local pt_mnist = import 'pytorch/nightly/mnist.libsonnet';

# Times in UTC
local schedules = {
  functional: '0 */12 * * *',
  convergence: '30 7 * * */1',
};

{
  [config.job_name + '.yaml']: std.manifestYamlDoc(config.cron_job(schedules[config.mode]))
    # Add model configs here
    for config in std.flattenArrays([
      tf_mnist.configs,
      tf_resnet_ctl.configs,
      tf_resnet_cfit.configs,
      pt_mnist.configs,
    ])
}
