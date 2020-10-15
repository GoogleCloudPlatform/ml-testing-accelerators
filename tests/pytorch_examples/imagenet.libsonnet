local common = import 'common.libsonnet';
local gpus = import 'templates/gpus.libsonnet';
local timeouts = import "templates/timeouts.libsonnet";

{
  local imagenet = common.PytorchTest {
    schedule: '0 3 * * 0',
    volumeMap+: {
      datasets: common.datasetsVolume
    },
    command: [
      'python3',
      'examples/imagenet/main.py',
      '/datasets/imagenet-mini',
    ],
    cpu: "13.0",
    memory: "40Gi",
  },
  local resnet50 = {
    modelName: 'resnet50',
    schedule: '0 4 * * 0',
    command+: [
      '--epochs=3',
      '--a=resnet50',
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 75.0,
          },
          comparison: "greater",
        },
      },
    },
  },
  local resnet18 = {
    modelName: 'resnet18',
    schedule: '0 5 * * 0',
    command+: [
      '--epochs=3',
      '--a=resnet18',
    ],
    regressionTestConfig+: {
      metric_success_conditions+: {
        "Accuracy/test_final": {
          success_threshold: {
            fixed_value: 65.0,
          },
          comparison: "greater",
        },
      },
    },
  },

  local v100 = {
    accelerator: gpus.teslaV100,
  },
  # NOTE: No special reason for the `6006` port. I think other free ports would
  # work too.
  local v100x4 = v100 {
    accelerator: gpus.teslaV100 + { count: 4 },
    command+: [
      '--dist-backend=nccl',
      '--dist-url=tcp://127.0.0.1:6006',
      '--multiprocessing-distributed',
      '--world-size=1',
      '--rank=0',
    ],
  },
  configs: [
    imagenet + resnet50 + v100 + timeouts.Hours(1),
    imagenet + resnet50 + v100x4 + timeouts.Hours(1),
    imagenet + resnet18 + v100 + timeouts.Hours(1),
  ],
}


