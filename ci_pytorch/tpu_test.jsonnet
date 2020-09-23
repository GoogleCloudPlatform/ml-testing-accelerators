local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";
local volumes = import "templates/volumes.libsonnet";

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
    requireTpuAvailableLabel: true,
  },
  accelerator: tpus.v3_8,

  volumeMap+: {
    dshm: volumes.MemoryVolumeSpec {
      name: "dshm",
      mountPath: "/dev/shm",
    },
  },

  cpu: "4.5",
  memory: "8Gi",

  command: utils.scriptCommand(
    |||
      echo "Running commands on GKE machine..."
      echo $XRT_TPU_CONFIG
      python pytorch/xla/test/test_train_mp_mnist.py
    |||
  ),
};

tputests.oneshotJob
