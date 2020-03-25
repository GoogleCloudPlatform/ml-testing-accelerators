from absl import logging
from absl.testing import absltest

import tensorflow as tf
import main
import metrics

class CloudMetricsHandlerTest(tf.test.TestCase):
  def setUp(self):
    self.temp_dir = self.create_tempdir().full_path
    self.summary_writer = tf.summary.create_file_writer(self.temp_dir)

    with self.summary_writer.as_default():
      tf.summary.scalar("foo", 1, 0)
      tf.summary.scalar("bar", tf.convert_to_tensor(1), 0)

      tf.summary.scalar("foo", 2, 100)
      tf.summary.scalar("bar", tf.convert_to_tensor(2), 100)

    self.summary_writer.flush()

  def tearDown(self):
    self.summary_writer.close()

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
      logger=logging, # pass in absl logging module
    )

    final_metrics = metrics_handler.get_metrics_from_events_dir()
    self.assertContainsSubset(['foo_final', 'foo_min', 'bar_final', 'bar_min'],
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
      logger=logging, # pass in absl logging module
    )

    final_metrics = metrics_handler.get_metrics_from_events_dir()
    with self.assertLogs(level='ERROR'):
      metrics_handler.compute_bounds_and_report_errors(
          {'final_status': 'success'},
          {'foo_final': [], 'total_wall_time': []},
          final_metrics
      )

if __name__ == '__main__':
  absltest.main()