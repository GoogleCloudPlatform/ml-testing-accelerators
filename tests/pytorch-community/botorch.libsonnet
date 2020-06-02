local basetest = import 'base.libsonnet';

local botorch = basetest.PytorchTest {
    schedule: '0 4 * * *',
    modelName: 'botorch',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/botorch/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

botorch.cronJob
