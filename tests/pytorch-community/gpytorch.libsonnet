local basetest = import 'base.libsonnet';

local gpytorch = basetest.PytorchTest {
    schedule: '0 10 * * *',
    modelName: 'gpytorch',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/gpytorch/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

gpytorch.cronJob
