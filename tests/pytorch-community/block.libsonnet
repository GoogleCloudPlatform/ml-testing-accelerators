local basetest = import 'base.libsonnet';

local block = basetest.PytorchTest {
    schedule: '0 3 * * *',
    modelName: 'block',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/block/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

block.cronJob
