local basetest = import 'base.libsonnet';

local cyclegan = basetest.PytorchTest {
    schedule: '0 5 * * *',
    modelName: 'cyclegan',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/cyclegan/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

cyclegan.cronJob
