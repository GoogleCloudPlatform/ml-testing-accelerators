import math

from absl.testing import absltest
from absl.testing import parameterized

import base
import metrics_pb2

class BaseCollectorTest(parameterized.TestCase):
  def setUp(self):
    self._collector = base.BaseCollector(
        metrics_pb2.TestCompletedEvent(benchmark_id="test_benchmark"), None)

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
    bounds = self._collector.compute_bounds("metric_key", assertion)
    # EQUAL is always inclusive
    want = base.Bounds(*expected_bounds,
        inclusive or comparison == 'EQUAL')
    self.assertSequenceAlmostEqual(bounds, want, places=3)

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
    base._get_historical_data_for_metric = lambda *args: value_history

    assertion = metrics_pb2.Assertion(
      std_devs_from_mean=metrics_pb2.Assertion.StdDevsFromMean(
        comparison=metrics_pb2.Assertion.Comparison.Value(comparison),
        std_devs=std_devs,
      ),
      inclusive_bounds=inclusive,
    )
    bounds = self._collector.compute_bounds("metric_key", assertion)
    want = base.Bounds(*expected_bounds, inclusive)
    self.assertSequenceAlmostEqual(bounds, want, places=3)

if __name__ == '__main__':
  absltest.main()
