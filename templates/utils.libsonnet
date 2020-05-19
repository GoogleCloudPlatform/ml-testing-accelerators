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
  // Formats multi-line script string into valid Kubernetes Container args.
  scriptCommand(script):
    [
      "/bin/bash",
      "-c",
      |||
        set -u
        set -e
        set -x
        
        %s
      ||| % script
    ],

  // Takes an object of the form {"test_name": test}, a default region, and an object of
  // the form {"region": accelerator}. Returns an object of the form
  // {"region/gen/test_name.yaml": cron_job_yaml} where region is either the region 
  // preferred for the accelerator used in the test (from regionAccelerators) or the
  // default region. Skips tests with schedule == null.
  // Use with jsonnet -S -m output_dir/ ...
  cronJobOutput(tests, defaultRegion, regionAccelerators={ }):
    local acceleratorRegions = std.foldl(
      function(result, region) result + {
        [accelerator.name]: region for accelerator in regionAccelerators[region]
      },
      std.objectFields(regionAccelerators),
      { },
    );
    local getDirectory(test) = (
      if std.objectHas(acceleratorRegions, test.accelerator.name) then
        acceleratorRegions[test.accelerator.name]
      else
        defaultRegion) + "/gen/";
    {
      [getDirectory(tests[name]) + name + ".yaml"]:
          std.manifestYamlDoc(tests[name].cronJob)
      for name in std.objectFields(tests) if tests[name].schedule != null
    }
}