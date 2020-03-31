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

from absl import logging
from absl.testing import absltest
import tensorflow as tf

import alert_handler
import main
import metrics

class CloudMetricsHandlerTest(absltest.TestCase):
  def setUp(self):
    self.logger = alert_handler.AlertHandler(
      project_id=None,
      write_to_logging=True,
      write_to_error_reporting=False,
      write_to_email=False)

    self.temp_dir = self.create_tempdir().full_path
    self.summary_writer = tf.summary.create_file_writer(self.temp_dir)

    with self.summary_writer.as_default():
      tf.summary.scalar("foo", 1, 0)
      tf.summary.scalar("bar", tf.convert_to_tensor(1), 0)

      tf.summary.scalar("foo", 2, 100)
      tf.summary.scalar("bar", tf.convert_to_tensor(2), 100)

    self.summary_writer.close()

    self.job_status_dict = {
      'job_status': 'SUCCESS',
      'stop_time': 1000,
      'start_time': 2000,
    }

  def test_get_metrics_from_event_dir(self):
    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      stackdriver_logs_link=None,
      metric_collection_config={
        'default_aggregation_strategies': ['final', 'min',]
      },
      regression_test_config={},
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    final_metrics = metrics_handler.get_metrics_from_events_dir(
        self.job_status_dict)
    self.assertContainsSubset(
        ['foo_final', 'foo_min', 'bar_final', 'bar_min', 'total_wall_time'],
        final_metrics.keys())

  def test_compute_bounds_and_report_errors_fixed_value(self):
    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      stackdriver_logs_link=None,
      metric_collection_config={
        'default_aggregation_strategies': ['final'],
        'tags_to_ignore': ['bar'],
      },
      regression_test_config={
        'metric_subset_to_alert': ['foo_final'],
        'metric_success_conditions': {
          'foo_final': {
            'success_threshold': {
              'fixed_value': 3.
            },
            'comparison': 'greater',
            'wait_for_n_points_of_history': 0,
          }
        }
      },
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    final_metrics = metrics_handler.get_metrics_from_events_dir(
        self.job_status_dict)
    with self.assertLogs(level='ERROR'):
      metrics_handler.compute_bounds_and_report_errors(
          {'foo_final': [], 'total_wall_time': []},
          final_metrics
      )

if __name__ == '__main__':
  absltest.main()
