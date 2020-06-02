local basetest = import 'base.libsonnet';

local skorch = basetest.PytorchTest {
    schedule: '0 18 * * *',
    modelName: 'skorch',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/skorch/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

skorch.cronJob
