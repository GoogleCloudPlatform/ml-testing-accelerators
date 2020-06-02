local basetest = import 'base.libsonnet';

local fastai = basetest.PytorchTest {
    schedule: '0 7 * * *',
    modelName: 'fastai',
    command: [
      'bash',
      'builder/test_community_repos/external_projects/fastai/run.sh',
    ],
    timeout: 3600, # 1 hour, in seconds.
};

fastai.cronJob
