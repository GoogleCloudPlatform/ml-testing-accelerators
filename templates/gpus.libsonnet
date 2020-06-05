# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

{
  GPUSpec:: {
    local gpu = self,

    name: "%(version)s-x%(number)d" % gpu,
    type: "gpu",
    version: error "Must specify GPUSpec `version`",
    number: 1,

    PodSpec:: {
      containerMap+: {
        train+: {
          resources+: {
            limits+: {
              "nvidia.com/gpu": gpu.number
            },
          },
        },
      },
      nodeSelector+: {
        "cloud.google.com/gke-accelerator": "nvidia-tesla-%(version)s" % gpu,
      },
    },
  },

  teslaK80: self.GPUSpec { version: "k80" },
  teslaV100: self.GPUSpec { version: "v100" },
}
