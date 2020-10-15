local common = import 'common.libsonnet';
local gpus = import 'templates/gpus.libsonnet';
local timeouts = import "templates/timeouts.libsonnet";

{
  local mnist = common.PytorchTest {
    schedule: '0 2 * * 0',
    modelName: 'mnist',
    command: [
      'python3',
      'examples/mnist/main.py',
    ],
    cpu: "13.0",
    memory: "40Gi",
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 98.0,
          },
          comparison: "greater",
        },
      },
    },
  },

  local v100 = {
    accelerator: gpus.teslaV100,
  },
  configs: [
    mnist + v100 + timeouts.Hours(1),
  ],
}


