local basetest = import 'base.libsonnet';

local opennmt = basetest.PytorchTest {
    schedule: '0 0 * * *',
    modelName: 'opennmt',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/OpenNMT/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

opennmt.cronJob
