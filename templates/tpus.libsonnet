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
  TpuSpec:: {
    local tpu = self,

    name: "v%(version)d-%(size)d" % tpu,
    version: error "Must override `type`",
    size: error "Must override `size`",
    preemptible: false,

    PodSpec:: {
      containerMap+: {
        train+: {
          resources+: {
            limits+: { [tpu.resource]: tpu.size },
          },
        },
      },
    },

    preemptiblePrefix:: if tpu.preemptible then
      "preemptible-"
    else
      "",
    resource:: "cloud-tpus.google.com/%(preemptiblePrefix)sv%(version)s" % tpu,
  },
  Preemptible:: {
    preemptible: true
  },

  v2_8: self.TpuSpec { version: 2, size: 8 },
  v3_8: self.TpuSpec { version: 3, size: 8 },
  v2_32: self.TpuSpec { version: 2, size: 32 },
  v3_32: self.TpuSpec { version: 3, size: 32 },
}
