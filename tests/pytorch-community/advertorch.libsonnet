local basetest = import 'base.libsonnet';

local advertorch = basetest.PytorchTest {
    schedule: '0 1 * * *',
    modelName: 'advertorch',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/advertorch/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

advertorch.cronJob
