local basetest = import 'base.libsonnet';

local pysyft = basetest.PytorchTest {
    schedule: '0 17 * * *',
    modelName: 'pysyft',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/pysyft/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

pysyft.cronJob
