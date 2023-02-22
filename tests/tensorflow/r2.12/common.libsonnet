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
local experimental = import 'experimental.libsonnet';
local metrics = import 'templates/metrics.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  ModelGardenTest:: common.ModelGardenTest {
    local config = self,

    frameworkPrefix: 'tf-r2.12.0',
    tpuSettings+: {
      softwareVersion: '2.11.0',
    },
    imageTag: 'r2.12.0',
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': {
            image: config.podTemplate.spec.containerMap.train.image,
            env+: [
              {
                name: 'TPU_NAME',
                valueFrom: {
                  fieldRef: {
                    fieldPath: "metadata.annotations['name.cloud-tpus.google.com/train']",
                  },
                },
              },
              {
                name: 'POD_UID',
                valueFrom: {
                  fieldRef: {
                    fieldPath: 'metadata.uid',
                  },
                },
              },
            ],
            local tpuCreateSettings = {
              acceleratorName: std.escapeStringBash(config.accelerator.name),
              softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
              startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
              sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
              testName: std.strReplace(config.testName, '.', '-'),
            },
            command: [
              'python3',
              '-c',
              |||
                import importlib_metadata
                import os
                import re
                import tensorflow as tf
                import urllib
                import json
                import cloud_tpu_client
                import sys
                print('python version: ' + str(sys.version))
                print('tf_version: ' + str(tf.__version__))
                print(str(tf.__file__))
                ctc = cloud_tpu_client.Client(tpu=os.path.basename('$(TPU_NAME)'), zone=os.path.dirname('$(TPU_NAME)'))
                ctc.wait_for_healthy()
                ctc.configure_tpu_version('2.12.0', restart_type='always')
                ctc.wait_for_healthy()
                _VERSION_SWITCHER_ENDPOINT = 'http://{}:8475/requestversion'
                url = _VERSION_SWITCHER_ENDPOINT.format(ctc.network_endpoints()[0]['ipAddress'])
                req = urllib.request.Request(url)
                resp = urllib.request.urlopen(req)
                version_details = json.loads(resp.read())
                print(version_details)
              |||,
            ],
          },
        },
      },
    },
  },
  tpuVm:: experimental.TensorFlowTpuVmMixin {
    local config = self,
    tpuSettings+: {
      softwareVersion: if config.accelerator.replicas == 1 then
        'tpu-vm-tf-2.11.0'
      else
        'tpu-vm-tf-2.11.0-pod',
    },
  },
  TfVisionTest:: self.ModelGardenTest + common.TfNlpVisionMixin {
    scriptConfig+: {
      runnerPath: 'official/vision/train.py',
    },
  },
  TfNlpTest:: self.ModelGardenTest + common.TfNlpVisionMixin {
    scriptConfig+: {
      runnerPath: 'official/nlp/train.py',
    },
  },
  local functional_schedule = '0 3 * * *',
  Functional:: mixins.Functional {
    schedule:
      if !(self.accelerator.type == 'tpu') || self.accelerator.name == 'v3-8' || self.accelerator.name == 'v4-8' then
        functional_schedule
      else
        null,
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
                  std_devs: 4.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
  // Override default schedule for Functional.
  RunNightly:: {
    schedule: '0 5 * * *',
  },
  Convergence:: mixins.Convergence {
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            examples_per_second: {
              AVERAGE: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'GREATER',
                  // TODO(wcromar): Tighten this restriction
                  std_devs: 2.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
}
