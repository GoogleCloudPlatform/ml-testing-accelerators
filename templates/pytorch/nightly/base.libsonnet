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

local base = import "../base.libsonnet";
local mixins = import "../../mixins.libsonnet";

{
  PyTorchTest:: base.PyTorchTest {
    frameworkPrefix: "pt-nightly",
    tpuVersion: "pytorch-nightly",
    imageTag: "nightly",
  },
  PyTorchPodTest:: base.PyTorchPodTest {
    frameworkPrefix: "pt-nightly",
    tpuVersion: "pytorch-nightly",
    imageTag: "nightly",
  },
  Functional:: mixins.Functional {
    # Run at 6AM PST daily.
    schedule: "0 14 * * *",
  },
  Convergence:: mixins.Convergence {
    # Run at 22:00 PST on Monday and Thursday.
    schedule: "0 6 * * 1,6",
  },
}
