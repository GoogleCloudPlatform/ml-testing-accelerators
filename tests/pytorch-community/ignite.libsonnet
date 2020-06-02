local basetest = import 'base.libsonnet';

local ignite = basetest.PytorchTest {
    schedule: '0 11 * * *',
    modelName: 'ignite',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/ignite/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

ignite.cronJob
