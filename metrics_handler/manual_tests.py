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

PROJECT_ID = 'xl-ml-test'
JOB_NAME = 'example-pt-imagenet-mini-gpu-1589155200'

# A collection of tests that read from live GCP resources. Replace args as
# necessary and then remove the leading underscore of the test name to run.
class ManualTests(absltest.TestCase):
  # To use this test, replace 'xl-ml-test' with your project name.
  def _test_compute_memory_metrics_job_does_not_exist(self):
    m = metrics.compute_memory_metrics(
        PROJECT_ID, 'tf-nightly-mnist-func-tesla-v100-x1-doesnotexist')
    self.assertFalse(m)


  # To use this test, replace PROJECT_ID with your GCP project name and
  # JOB_NAME with a recent job that ran. Change the assert if your job
  # does not use GPUs.
  def _test_compute_memory_metrics(self):
    m = metrics.compute_memory_metrics(PROJECT_ID, JOB_NAME)
    self.assertTrue('vm_memory_usage_bytes' in m and \
                    m['vm_memory_usage_bytes'].metric_value > 0)
    self.assertTrue('gpu_memory_usage_bytes' in m and \
                    m['gpu_memory_usage_bytes'].metric_value > 0)

  # To use this test, replace PROJECT_ID with your GCP project name and
  # JOB_NAME with a recent job that ran.
  def _test_get_computed_metrics_with_memory_metrics(self):
    job_status_dict = {
        'start_time': 0,
        'stop_time': 20,
    }
    raw_metrics = {
      'accuracy': [
        metrics.MetricPoint(metric_value=.2, wall_time=0),
        metrics.MetricPoint(metric_value=.4, wall_time=10),
        metrics.MetricPoint(metric_value=.6, wall_time=20),
      ],
      'other_key': [
        metrics.MetricPoint(metric_value=.8, wall_time=0),
      ],
    }
    tta_config = {
        'accuracy_tag': 'accuracy',
        'accuracy_threshold': 0.4,
    }
    computed_metrics = metrics.get_computed_metrics(
        raw_metrics, job_status_dict, tta_config=tta_config,
        project_id=PROJECT_ID, job_name=JOB_NAME, find_memory_metrics=True)
    self.assertContainsSubset(
        ['total_wall_time', 'time_to_accuracy', 'vm_memory_usage_bytes'],
        computed_metrics.keys())




if __name__ == '__main__':
  absltest.main()
