local basetest = import 'base.libsonnet';

local allennlp = basetest.PytorchTest {
    schedule: '0 2 * * *',
    modelName: 'allennlp',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/allennlp/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

allennlp.cronJob
