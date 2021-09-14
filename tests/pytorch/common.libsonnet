// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local common = import '../common.libsonnet';
local metrics = import 'templates/metrics.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  local PyTorchBaseTest = common.CloudAcceleratorTest {
    configMaps+: ['pytorch-nfs-ip'],

    metricConfig: metrics.MetricCollectionConfigHelper {
      sourceMap:: {
        tensorboard: metrics.TensorBoardSourceHelper {
          aggregateAssertionsMap+:: {
            ExecuteTime__Percentile_99_sec: {
              FINAL: {
                std_devs_from_mean: {
                  comparison: 'LESS',
                  std_devs: 5.0,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 10,
              },
            },
            aten_ops_sum: {
              FINAL: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'LESS',
                  std_devs: 0.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
          exclude_tags: [
            'LearningRate',
          ],
          include_tags: [
            {
              strategies: [
                'FINAL',
              ],
              tag_pattern: '*',
            },
          ],
          merge_runs: true,
        },
        literals: {
          assertions: {
            duration: {
              inclusive_bounds: false,
              std_devs_from_mean: {
                comparison: 'LESS',
                std_devs: 5,
              },
              wait_for_n_data_points: 10,
            },
          },
        },
      },
    },
  },
  PyTorchTest:: PyTorchBaseTest {
    local config = self,

    image: 'gcr.io/xl-ml-test/pytorch-xla',
    volumeMap+: {
      dshm: volumes.MemoryVolumeSpec {
        name: 'dshm',
        mountPath: '/dev/shm',
      },
    },

    cpu: '4.5',
    memory: '8Gi',

    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              XLA_USE_BF16: '0',
            },
          },
        },
      },
    },
  },
  PyTorchGkePodTest:: PyTorchBaseTest {
    local config = self,

    image: 'gcr.io/xl-ml-test/pytorch-xla',
    // Resources for created workers.
    workerCpu: '4',
    workerMemory: '4Gi',
    // These should _only_ be PersistentVolumeClaims. Defaults to
    // config.volumeMap, which also mounts on the coordinator. To mount a volume
    // on just the workers, directly override config.workerVolumes.
    workerVolumes: config.volumeMap,

    jobTemplate+:: {
      spec+: {
        backoffLimit: 0,
        template+: {
          spec+: {
            serviceAccountName: 'pytorch-xla-pods',
            containerMap+: {
              train+: {
                // Use `image` and `imageTag` for workers instead.
                image: 'gcr.io/xl-ml-test/pytorch-pods:nightly',
                // Override the Docker ENTRYPOINT.
                command: [
                  'python3',
                  'launch_k8s_workers.py',
                  '--name=$(JOB_NAME)',
                  '--image=%s:%s' % [config.image, config.imageTag],
                  '--owner_name=$(POD_NAME)',
                  '--owner_uid=$(POD_UID)',
                  '--namespace=$(POD_NAMESPACE)',
                  '--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)',
                  '--cpu=%s' % config.workerCpu,
                  '--memory=%s' % config.workerMemory,
                  '--volumes=%s' % std.join(',',
                                            [
                                              '%(name)s:%(mountPath)s' % config.workerVolumes[k]
                                              for k in std.objectFields(config.workerVolumes)
                                            ]),
                  '--',
                  // config.args is distributed to the workers.
                ],
              },
            },
          },
        },
      },
    },
  },
  // Use `torch_xla.distributed.xla_dist to create an instance group of client
  // workers for a TPU pod.
  PyTorchXlaDistPodTest:: PyTorchBaseTest {
    local config = self,

    image: 'gcr.io/xl-ml-test/pytorch-pods',
    instanceType: 'n1-standard-8',
    condaEnv: 'torch-xla-nightly',
    xlaDistFlags: '',

    jobTemplate+:: {
      spec+: {
        backoffLimit: 0,
        template+: {
          spec+: {
            containerMap+: {
              train+: {
                envMap+: {
                  MACHINE_TYPE: config.instanceType,
                  ACCELERATOR_TYPE: config.acceleratorName,
                  CONDA_ENV: config.condaEnv,
                  XLA_DIST_FLAGS: config.xlaDistFlags,
                },
              },
            },
          },
        },
      },
    },
  },
}
