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
    frameworkPrefix: "pt-pytorch-1.5",
    tpuVersion: "pytorch-1.5",
    imageTag: "r1.5",
  },
  Functional:: mixins.Functional {
    # Run at 8AM PST daily.
    schedule: "0 16 * * *",
  },
  Convergence:: mixins.Convergence {
    # Run twice per week.
    schedule: "0 20 * * 1,6",
  },
}
