local basetest = import 'base.libsonnet';

local pyro = basetest.PytorchTest {
    schedule: '0 16 * * *',
    modelName: 'pyro',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/pyro/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

pyro.cronJob
