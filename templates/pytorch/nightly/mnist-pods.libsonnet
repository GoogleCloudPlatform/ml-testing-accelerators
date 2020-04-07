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

local base = import "base.libsonnet";
local mixins = import "../../mixins.libsonnet";
local timeouts = import "../../timeouts.libsonnet";
local tpus = import "../../tpus.libsonnet";
local utils = import "../../utils.libsonnet";

{
  local mnist = base.PyTorchPodTest {
    image: "gcr.io/xl-ml-test/pytorch-pods",
    imageTag: "latest",
    modelName: "mnist-pods",
    command: [
      "python /usr/share/torch-xla-nightly/pytorch/xla/test/test_train_mp_mnist.py",
    ],
    
    accelerator: tpus.v3_32,
  },
  local convergence = base.Convergence {
    regressionTestConfig: null,
  },
  configs: [
    mnist + convergence + mixins.Experimental,
  ],
}
