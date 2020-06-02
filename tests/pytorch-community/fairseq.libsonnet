local basetest = import 'base.libsonnet';

local fairseq = basetest.PytorchTest {
    schedule: '0 6 * * *',
    modelName: 'fairseq',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/fairseq/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

fairseq.cronJob
