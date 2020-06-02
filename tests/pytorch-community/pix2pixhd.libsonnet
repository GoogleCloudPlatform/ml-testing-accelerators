local basetest = import 'base.libsonnet';

local pix2pixhd = basetest.PytorchTest {
    schedule: '0 15 * * *',
    modelName: 'pix2pixhd',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/pix2pixHD/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

pix2pixhd.cronJob
