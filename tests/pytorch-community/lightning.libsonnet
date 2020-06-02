local basetest = import 'base.libsonnet';

local lightning = basetest.PytorchTest {
    schedule: '0 12 * * *',
    modelName: 'lightning',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/lightning/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

lightning.cronJob
