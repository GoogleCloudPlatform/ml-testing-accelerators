local basetest = import 'base.libsonnet';

local pennylane = basetest.PytorchTest {
    schedule: '0 14 * * *',
    modelName: 'pennylane',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/pennylane/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

pennylane.cronJob
