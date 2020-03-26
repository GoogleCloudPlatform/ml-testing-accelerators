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

import math

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

import metrics


class MetricsTest(parameterized.TestCase):
  def setUp(self):
    self.temp_dir = self.create_tempdir().full_path
    self.summary_writer = tf.summary.create_file_writer(self.temp_dir)

    with self.summary_writer.as_default():
      tf.summary.scalar("foo", 1, 0)
      tf.summary.scalar("accuracy", .125, 0)

      tf.summary.scalar("foo", 2, 100)
      tf.summary.scalar("accuracy", .5, 100)

      tf.summary.scalar("accuracy", .25, 200)

    self.summary_writer.flush()

  def tearDown(self):
    self.summary_writer.close()

  def test_read_metrics_from_event_dir(self):
    raw_metrics = metrics.read_metrics_from_events_dir(self.temp_dir)

    self.assertLen(raw_metrics['foo'], 2)
    self.assertLen(raw_metrics['accuracy'], 3)

  def test_read_metrics_from_event_dir_with_ignore(self):
    raw_metrics = metrics.read_metrics_from_events_dir(self.temp_dir, {'foo'})

    self.assertNotIn('foo', raw_metrics)
    self.assertLen(raw_metrics['accuracy'], 3)

  def test_aggregate_metrics_default_all(self):
    raw_metrics = metrics.read_metrics_from_events_dir(self.temp_dir)
    final_metrics = metrics.aggregate_metrics(
      raw_metrics, default_strategies=['final', 'min', 'max'])

    # Remove wall time, since it's non-deterministic
    metric_to_value = {
      metric_name: point.metric_value 
      for metric_name, point in final_metrics.items()
    }

    self.assertDictEqual(
      metric_to_value,
      {
        'foo_final': 2,
        'foo_min': 1,
        'foo_max': 2,
        'accuracy_final': .25,
        'accuracy_min': .125,
        'accuracy_max': .5,
      }
    )

  def test_aggregate_metrics_custom(self):
    raw_metrics = metrics.read_metrics_from_events_dir(self.temp_dir)
    final_metrics = metrics.aggregate_metrics(
      raw_metrics,
      default_strategies=['final', 'min', 'max'],
      metric_strategies={'accuracy': ['max']}
    )

    # Remove wall time, since it's non-deterministic
    metric_to_value = {
      metric_name: point.metric_value 
      for metric_name, point in final_metrics.items()
    }

    self.assertDictEqual(
      metric_to_value,
      {
        'foo_final': 2,
        'foo_min': 1,
        'foo_max': 2,
        'accuracy_max': .5,
      }
    )

  def test_total_wall_time(self):
    raw_metrics = {
      'foo': [
        metrics.MetricPoint(metric_value=0, wall_time=0),
        metrics.MetricPoint(metric_value=1, wall_time=1),
      ],
      'bar': [
        metrics.MetricPoint(metric_value=0, wall_time=0),
        metrics.MetricPoint(metric_value=2, wall_time=2),
      ],
    }

    total_wall_time = metrics.total_wall_time(raw_metrics)
    self.assertEqual(
        total_wall_time, metrics.MetricPoint(metric_value=2, wall_time=2))

  def test_time_to_accuracy(self):
    raw_metrics = {
      'accuracy': [
        metrics.MetricPoint(metric_value=.2, wall_time=0),
        metrics.MetricPoint(metric_value=.4, wall_time=10),
        metrics.MetricPoint(metric_value=.6, wall_time=20),
      ],
      'other_metric': [
        metrics.MetricPoint(metric_value=1, wall_time=15)
      ]
    }

    time_to_accuracy = metrics.time_to_accuracy(
        raw_metrics, tag='accuracy', threshold=.4)
    self.assertEqual(time_to_accuracy,
        metrics.MetricPoint(metric_value=10, wall_time=10))

  @parameterized.named_parameters(
    ('equal', 'equal', 5., (5., 5.)),
    ('less', 'less', 5., (-math.inf, 5.)),
    ('less_or_equal', 'less_or_equal', 5., (-math.inf, 5.)),
    ('greater', 'greater', 5., (5., math.inf)),
  )
  def test_metric_bounds_fixed_value(self, comparison, threshold_value, expected_bounds):
    # `fixed_value` comparison ignores value history
    value_history = []
    threshold = metrics.Threshold('fixed_value', threshold_value)
    bounds = metrics.metric_bounds(value_history, threshold, comparison)

    self.assertSequenceAlmostEqual(bounds, expected_bounds)

  @parameterized.named_parameters(
    ('less_1_dev', 'less', 1, (-math.inf, 4.414)),
    ('less_or_equal', 'less', 1, (-math.inf, 4.414)),
    ('less_2_dev', 'less', 2, (-math.inf, 5.828)),
    ('greater', 'greater', 1, (1.586, math.inf)),
  )
  def test_metric_bounds_stddevs_from_mean(self, comparison, threshold_value, expected_bounds):
    # mean = 3, stddev = ~1.414
    value_history = [
      metrics.MetricPoint(metric_value=1, wall_time=10),
      metrics.MetricPoint(metric_value=2, wall_time=20),
      metrics.MetricPoint(metric_value=3, wall_time=30),
      metrics.MetricPoint(metric_value=4, wall_time=40),
      metrics.MetricPoint(metric_value=5, wall_time=50),
    ]

    threshold = metrics.Threshold('stddevs_from_mean', threshold_value)
    bounds = metrics.metric_bounds(value_history, threshold, comparison)

    self.assertSequenceAlmostEqual(bounds, expected_bounds, places=3)

  @parameterized.named_parameters(
    ('less', 5., (-math.inf, 10.), False, True),
    ('equal', 5., (5., 5.), True, True),
    ('less_inclusive', 5., (-math.inf, 5.), True, True),
    ('within', 3., (0., 5.), False, True),
    ('less_exclusive', 5., (-math.inf, 5.), False, False),
    ('greater', 5., (-math.inf, 5.), False, False),
    ('outside', 10, (0., 5.), False, False),
  )
  def test_within_bounds(self, value, bounds, inclusive, expected):
    within_bounds = metrics.within_bounds(value, *bounds, inclusive)
    self.assertIs(within_bounds, expected)


if __name__ == '__main__':
  absltest.main()
