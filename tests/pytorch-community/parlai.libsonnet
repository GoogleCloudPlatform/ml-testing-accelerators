local basetest = import 'base.libsonnet';

local parlai = basetest.PytorchTest {
    schedule: '0 13 * * *',
    modelName: 'parlai',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/parlai/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

parlai.cronJob
