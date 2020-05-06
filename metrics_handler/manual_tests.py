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

from absl.testing import absltest

import metrics

# A collection of tests that read from live GCP resources. Replace args as
# necessary and then remove the leading underscore of the test name to run.
class ManualTests(absltest.TestCase):
  # To use this test, replace 'xl-ml-test' with your proejct name.
  def _test_compute_memory_metrics_job_does_not_exist(self):
    m = {}
    metrics.compute_memory_metrics(
        m, 'xl-ml-test', 'tf-nightly-mnist-func-tesla-v100-x1-doesnotexist')
    self.assertFalse(m)


  # To use this test, replace the args to compute_memory_metrics with your
  # project ID and the name of a recent job. Change the assert if your job
  # does not use GPUs.
  def test_compute_memory_metrics(self):
    m = {}
    metrics.compute_memory_metrics(
        m, 'xl-ml-test', 'example-pt-imagenet-mini-gpu-manual-fhjhh')
    print(m)
    self.assertTrue('vm_memory_usage_bytes' in m and \
                    m['vm_memory_usage_bytes'].metric_value > 0)
    self.assertTrue('gpu_memory_usage_bytes' in m and \
                    m['gpu_memory_usage_bytes'].metric_value > 0)


if __name__ == '__main__':
  absltest.main()
