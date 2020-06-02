local basetest = import 'base.libsonnet';

local flair = basetest.PytorchTest {
    schedule: '0 8 * * *',
    modelName: 'flair',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/flair/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

flair.cronJob
