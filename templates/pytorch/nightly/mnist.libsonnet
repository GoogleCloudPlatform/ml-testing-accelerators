local jobs = import 'jobs.libsonnet';
local modes = import '../../modes.libsonnet';
local timeouts = import '../../timeouts.libsonnet';
local tpus = import '../../tpus.libsonnet';

{
  local mnist = jobs.PyTorchJobConfig {
    model_name: 'mnist-pytorch',
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_mnist.py',
    ],
  },
  local convergence = modes.Convergence {
    accelerator+: tpus.Preemptible,
  },
  local v2_8 = {
    accelerator+: tpus.v2_8,
  },
  local v3_8 = {
    accelerator+: tpus.v3_8,
  },

  configs: [
    mnist + v2_8 + convergence + timeouts.hours(1),
    mnist + v3_8 + convergence + timeouts.hours(1),
  ],
}
