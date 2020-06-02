local basetest = import 'base.libsonnet';

local geometric = basetest.PytorchTest {
    schedule: '0 9 * * *',
    modelName: 'geometric',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/geometric/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

geometric.cronJob
