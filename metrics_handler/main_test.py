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

import os
import pathlib
import json

from absl import logging
from absl.testing import absltest
import tensorflow as tf

import alert_handler
import job_status_handler
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
      tf.summary.scalar("bar", tf.convert_to_tensor(1), 100)

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
      debug_info=None,
      metric_collection_config={
        'default_aggregation_strategies': ['final', 'min',]
      },
      regression_test_config={},
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    _, aggregated_metrics = metrics_handler.get_metrics_from_events_dir()
    self.assertContainsSubset(
        ['foo_final', 'foo_min', 'bar_final', 'bar_min'],
        aggregated_metrics.keys())

  def test_get_metrics_from_perfzero_summary(self):
    summary_dir = os.path.join(self.temp_dir, 'date_and_time')
    pathlib.Path(summary_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'perfzero_summary.json')
    with open(summary_path, 'w') as f:
      json.dump({
        "execution_id": "execution_id",
        "execution_timestamp": 1234567890.1,
        "benchmark_result": {
          "wall_time": 1234,
          "metrics": [
            {
              "name": "exp_per_second",
              "value": 1.1,
            },
            {
              "name": "avg_exp_per_second",
              "value": 2.2,
            },
            {
              "name": "startup_time",
              "value": 3.3
            }
          ],
        },
        "benchmark_info": {
          "not": "important",
        },
        "setup_info": {},
        "ml_framework_info": {
          "not": "important",
        },
        "system_info": {
          "not": "important"
        },
        "process_info": {
          "max_rss": 4.4,
          "max_vms": 5.5,
          "max_cpu_percent": 6.6,
        }
        }, f)

    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      debug_info=None,
      metric_collection_config={},
      regression_test_config={},
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    aggregated_metrics = metrics_handler.get_metrics_from_perfzero_summary()
    self.assertDictEqual(
      {
        "total_wall_time": metrics.MetricPoint(1234, 1234567890.1),
        "exp_per_second": metrics.MetricPoint(1.1, 1234567890.1),
        "avg_exp_per_second": metrics.MetricPoint(2.2, 1234567890.1),
        "startup_time": metrics.MetricPoint(3.3, 1234567890.1),
        "max_rss": metrics.MetricPoint(4.4, 1234567890.1),
        "max_vms": metrics.MetricPoint(5.5, 1234567890.1),
        "max_cpu_percent": metrics.MetricPoint(6.6, 1234567890.1),
      },
      aggregated_metrics
    )

  def test_get_metrics_from_perfzero_summary_not_found(self):
    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      debug_info=None,
      metric_collection_config={},
      regression_test_config={},
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    aggregated_metrics = metrics_handler.get_metrics_from_perfzero_summary()
    self.assertEmpty(aggregated_metrics)

  def test_compute_bounds_and_report_errors_fixed_value(self):
    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      debug_info=None,
      metric_collection_config={
        'default_aggregation_strategies': ['final'],
        'tags_to_ignore': ['bar'],
      },
      regression_test_config={
        'metric_subset_to_alert': ['foo_final'],
        'required_metrics': ['foo_final'],
        'metric_success_conditions': {
          'foo_final': {
            'success_threshold': {
              'fixed_value': 3.
            },
            'comparison': 'greater',
            'wait_for_n_points_of_history': 0,
          },
        },
      },
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    _, aggregated_metrics = metrics_handler.get_metrics_from_events_dir()
    with self.assertLogs(level='ERROR'):
      metrics_handler.compute_bounds_and_report_errors(
          {'foo_final': [], 'total_wall_time': []},
          aggregated_metrics, job_status_handler.SUCCESS
      )
    # No error should be logged for out-of-bounds metrics if the job failed.
    with self.assertRaises(AssertionError):
      with self.assertLogs(level='ERROR'):
        metrics_handler.compute_bounds_and_report_errors(
            {'foo_final': [], 'total_wall_time': []},
            aggregated_metrics, job_status_handler.FAILURE
        )

  def test_compute_bounds_and_report_errors_missing_metric(self):
    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      debug_info=None,
      metric_collection_config={
        'default_aggregation_strategies': ['final'],
        'tags_to_ignore': ['bar'],
      },
      regression_test_config={
        'metric_subset_to_alert': ['foo_final'],
        'required_metrics': ['foo_final', 'fake_metric'],
        'metric_success_conditions': {
          'foo_final': {
            # foo_final=2.0 from the setUp() method above.
            'success_threshold': {
              'fixed_value': 1.
            },
            'comparison': 'greater',
            'wait_for_n_points_of_history': 0,
          },
        },
      },
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    _, aggregated_metrics = metrics_handler.get_metrics_from_events_dir()
    # Metrics are within bounds but are missing the `fake_metric`, which
    # is in `required_metrics`, so an error should be created.
    with self.assertLogs(level='ERROR'):
      metrics_handler.compute_bounds_and_report_errors(
          {'foo_final': [], 'total_wall_time': []},
          aggregated_metrics, job_status_handler.SUCCESS
      )
    # Error should be logged for missing required_metrics even if the test
    # run ended in failure.
    with self.assertLogs(level='ERROR'):
      metrics_handler.compute_bounds_and_report_errors(
          {'foo_final': [], 'total_wall_time': []},
          aggregated_metrics, job_status_handler.FAILURE
      )

  def test_compute_bounds_and_report_errors_stddevs(self):
    metrics_handler = main.CloudMetricsHandler(
      test_name="test",
      events_dir=self.temp_dir,
      debug_info=None,
      metric_collection_config={
        'default_aggregation_strategies': ['final'],
        'tags_to_ignore': ['foo'],
      },
      regression_test_config={
        'metric_subset_to_alert': ['bar_final'],
        'metric_success_conditions': {
          'bar_final': {
            'success_threshold': {
              'stddevs_from_mean': 1.
            },
            'comparison': 'greater_or_equal',
            'wait_for_n_points_of_history': 0,
          },
        },
      },
      test_type=None,
      accelerator=None,
      framework_version=None,
      logger=self.logger,
    )

    _, aggregated_metrics = metrics_handler.get_metrics_from_events_dir()
    # Average is higher than current value - this should trigger an alert.
    with self.assertLogs(level='ERROR'):
      metrics_handler.compute_bounds_and_report_errors(
          {'bar_final': [metrics.MetricPoint(10.0, 111),
                         metrics.MetricPoint(10.0, 112),
                         metrics.MetricPoint(10.0, 113)],
           'total_wall_time': []},
          aggregated_metrics, job_status_handler.SUCCESS
      )
    # No error should be logged for out-of-bounds metrics if the job failed.
    with self.assertRaises(AssertionError):
      with self.assertLogs(level='ERROR'):
        metrics_handler.compute_bounds_and_report_errors(
            {'bar_final': [metrics.MetricPoint(10.0, 111),
                           metrics.MetricPoint(10.0, 112),
                           metrics.MetricPoint(10.0, 113)],
             'total_wall_time': []},
            aggregated_metrics, job_status_handler.FAILURE
        )
    # Average == current value - this should not trigger an alert since
    # we are using `greater_or_equal`.
    with self.assertRaises(AssertionError):
      with self.assertLogs(level='ERROR'):
        metrics_handler.compute_bounds_and_report_errors(
            {'bar_final': [metrics.MetricPoint(1.0, 111),
                           metrics.MetricPoint(1.0, 112),
                           metrics.MetricPoint(1.0, 113)],
             'total_wall_time': []},
            aggregated_metrics, job_status_handler.SUCCESS
        )


if __name__ == '__main__':
  absltest.main()
