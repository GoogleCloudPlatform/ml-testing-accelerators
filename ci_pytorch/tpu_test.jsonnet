local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'ci-pt',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 900, # 15 minutes, in seconds.

  image: std.extVar('image'),
  imageTag: std.extVar('image-tag'),

  tpuSettings+: {
    softwareVersion: 'pytorch-nightly',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      echo "before testings"
      echo $XRT_TPU_CONFIG
      python3 pytorch/xla/test/test_train_mp_mnist.py
    |||
  ),
};

tputests.oneshotJob
