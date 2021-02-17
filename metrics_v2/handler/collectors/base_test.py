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

import dataclasses
import datetime
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2

import base
from handler import bigquery_client
from handler import utils
import metrics_pb2

class BaseCollectorTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ('no_bounds', -math.inf, math.inf, (-math.inf, math.inf)),
    ('upper', -math.inf, 200, (-math.inf, 200)),
    ('lower', 100, math.inf, (100, math.inf)),
    ('upper_and_lower', 100, 200, (100, 200), True),
  )
  def test_compute_bounds_within_bounds(
      self, lower, upper, expected_bounds, inclusive=False):
    assertion = metrics_pb2.Assertion(
      within_bounds=metrics_pb2.Assertion.WithinBounds(
        lower_bound=lower,
        upper_bound=upper,
      ),
      inclusive_bounds=inclusive,
    )
    collector = base.BaseCollector(
        metrics_pb2.TestCompletedEvent(benchmark_id="test_benchmark"),
        None)
    bounds = collector.compute_bounds("metric_key", assertion)
    self.assertEqual(bounds, utils.Bounds(*expected_bounds,
        inclusive))

  @parameterized.named_parameters(
    ('equal', 'EQUAL', 5., (5., 5.)),
    ('less', 'LESS', 5., (-math.inf, 5.)),
    ('less_or_equal', 'LESS', 5., (-math.inf, 5.), True),
    ('greater', 'GREATER', 5., (5., math.inf)),
  )
  def test_compute_bounds_fixed_value(
      self, comparison, threshold_value, expected_bounds, inclusive=False):

    assertion = metrics_pb2.Assertion(
      fixed_value=metrics_pb2.Assertion.FixedValue(
        comparison=metrics_pb2.Assertion.Comparison.Value(comparison),
        value=threshold_value,
      ),
      inclusive_bounds=inclusive,
    )
    collector = base.BaseCollector(
        metrics_pb2.TestCompletedEvent(benchmark_id="test_benchmark"), None)
    bounds = collector.compute_bounds("metric_key", assertion)
    self.assertSequenceAlmostEqual(
        dataclasses.astuple(bounds),
        # EQUAL is always inclusive
        dataclasses.astuple(utils.Bounds(*expected_bounds,
            inclusive or comparison == 'EQUAL')),
        places=3)

  @parameterized.named_parameters(
    ('less', 'LESS', 5, .20, (-math.inf, 6)),
    ('less_or_equal', 'LESS', 5, .20, (-math.inf, 6), True),
    ('greater', 'GREATER', 5, .2, (4, math.inf)),
    ('greater_or_equal', 'GREATER', 5, .2, (4, math.inf), True),
    ('within', 'WITHIN', 5, .2, (4, 6)),
  )
  def test_compute_bounds_percent_difference_with_target_value(
      self, comparison, target, pct_diff, expected_bounds, inclusive=False):
    assertion = metrics_pb2.Assertion(
      percent_difference=metrics_pb2.Assertion.PercentDifference(
        comparison=metrics_pb2.Assertion.Comparison.Value(comparison),
        value=target,
        percent=pct_diff,
      ),
      inclusive_bounds=inclusive,
    )
    collector = base.BaseCollector(
        metrics_pb2.TestCompletedEvent(benchmark_id="test_benchmark"),
        None)
    bounds = collector.compute_bounds("metric_key", assertion)
    self.assertEqual(bounds, utils.Bounds(*expected_bounds,
        inclusive))

  @parameterized.named_parameters(
    ('less', 'LESS', 1/3, (-math.inf, 4)),
    ('less_or_equal', 'LESS', 1/3, (-math.inf, 4), True),
    ('greater', 'GREATER', 1/3, (2, math.inf)),
    ('greater_or_equal', 'GREATER', 1/3, (2, math.inf), True),
    ('within', 'WITHIN', 1/3, (2, 4)),
  )
  def test_compute_bounds_percent_difference_with_mean_value(
      self, comparison, pct_diff, expected_bounds, inclusive=False):
    # mean = 3
    value_history = range(1, 6)
    class _FakeMetricStore:
      def get_metric_history(*args, **kwargs):
        return [
          bigquery_client.MetricHistoryRow(
              '', '', datetime.datetime.now(), '', v)
          for v in value_history]

    assertion = metrics_pb2.Assertion(
      percent_difference=metrics_pb2.Assertion.PercentDifference(
        comparison=metrics_pb2.Assertion.Comparison.Value(comparison),
        use_historical_mean=True,
        percent=pct_diff,
      ),
      inclusive_bounds=inclusive,
    )
    collector = base.BaseCollector(
        metrics_pb2.TestCompletedEvent(benchmark_id="test_benchmark"),
        None,
        _FakeMetricStore())
    bounds = collector.compute_bounds("metric_key", assertion)
    self.assertSequenceAlmostEqual(
        dataclasses.astuple(bounds),
        dataclasses.astuple(utils.Bounds(*expected_bounds, inclusive)),
        places=3)

  @parameterized.named_parameters(
    ('less', 'LESS', 1, (-math.inf, 4.414)),
    ('less_or_equal', 'LESS', 1, (-math.inf, 4.414), True),
    ('greater', 'GREATER', 1, (1.586, math.inf)),
    ('greater_or_equal', 'GREATER', 1, (1.586, math.inf), True),
    ('less_2_dev', 'LESS', 2, (-math.inf, 5.828)),
    ('within', 'WITHIN', 1, (1.586, 4.414)),
  )
  def test_compute_bounds_stddevs_from_mean(
      self, comparison, std_devs, expected_bounds, inclusive=False):
    # mean = 3, stddev = ~1.414
    value_history = range(1, 6)
    class _FakeMetricStore:
      def get_metric_history(*args, **kwargs):
        return [
          bigquery_client.MetricHistoryRow(
              '', '', datetime.datetime.now(), '', v)
          for v in value_history]

    assertion = metrics_pb2.Assertion(
      std_devs_from_mean=metrics_pb2.Assertion.StdDevsFromMean(
        comparison=metrics_pb2.Assertion.Comparison.Value(comparison),
        std_devs=std_devs,
      ),
      inclusive_bounds=inclusive,
    )
    collector = base.BaseCollector(
        metrics_pb2.TestCompletedEvent(benchmark_id="test_benchmark"),
        None,
        _FakeMetricStore())
    bounds = collector.compute_bounds("metric_key", assertion)
    self.assertSequenceAlmostEqual(
        dataclasses.astuple(bounds),
        # EQUAL is always inclusive
        dataclasses.astuple(utils.Bounds(*expected_bounds,
            inclusive or comparison == 'EQUAL')),
        places=3)

  @parameterized.named_parameters(
    ('none', None, None, datetime.datetime.fromtimestamp(0)),
    ('window', datetime.timedelta(days=2), None, datetime.datetime(2021, 2, 14, 0, 0, 0)),
    ('timestamp', None, datetime.datetime(2021, 2, 14, 0, 0, 0), datetime.datetime(2021, 2, 14, 0, 0, 0)),
    ('both_1', datetime.timedelta(days=3), datetime.datetime(2021, 2, 14, 0, 0, 0), datetime.datetime(2021, 2, 14, 0, 0, 0)),
    ('both_2', datetime.timedelta(days=2), datetime.datetime(2021, 2, 13, 0, 0, 0), datetime.datetime(2021, 2, 14, 0, 0, 0)),
  )
  def test_min_time(self, window, timestamp, expected_min):
    start_time = timestamp_pb2.Timestamp()
    start_time.FromDatetime(datetime.datetime(2021, 2, 16, 0, 0, 0))

    min_timestamp = timestamp_pb2.Timestamp()
    if timestamp:
      min_timestamp.FromDatetime(timestamp)

    time_window = duration_pb2.Duration()
    if window:
      time_window.FromTimedelta(window)

    assertion = metrics_pb2.Assertion(
      std_devs_from_mean=metrics_pb2.Assertion.StdDevsFromMean(
        comparison=metrics_pb2.Assertion.Comparison.WITHIN,
        std_devs=1,
      ),
      time_window=time_window,
      min_timestamp=min_timestamp,
    )

    metric_store = bigquery_client.BigQueryMetricStore('fake_dataset', 'fake_project')
    metric_store.get_metric_history = mock.Mock(return_value=[])

    collector = base.BaseCollector(
      metrics_pb2.TestCompletedEvent(
        benchmark_id="test_benchmark",
        start_time=start_time,
      ),
      None,
      metric_store)

    collector.compute_bounds("metric_key", assertion)

    metric_store.get_metric_history.assert_called_with(
      benchmark_id="test_benchmark",
      metric_key="metric_key",
      min_time=expected_min)


if __name__ == '__main__':
  absltest.main()
