import dataclasses
import datetime
import math

from absl.testing import absltest
from absl.testing import parameterized

from handler import bigquery_client
from handler import utils
from handler.collectors import base
import metrics_pb2

class BaseCollectorTest(parameterized.TestCase):
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
    value_history = list(range(1, 6))
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

if __name__ == '__main__':
  absltest.main()