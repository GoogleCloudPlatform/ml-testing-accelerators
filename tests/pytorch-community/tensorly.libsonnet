local basetest = import 'base.libsonnet';

local tensorly = basetest.PytorchTest {
    schedule: '0 19 * * *',
    modelName: 'tensorly',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/tensorly/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

tensorly.cronJob
