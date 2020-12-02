import math

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import duration_pb2

from handler.collectors import base
from handler.collectors import literal_collector

import metrics_pb2

class LiteralCollectorTest(parameterized.TestCase):
  def test_assert_duration(self):
    metric_source = metrics_pb2.MetricSource(
      literals=metrics_pb2.LiteralSource(
        assertions={
          "duration": metrics_pb2.Assertion(
            within_bounds=metrics_pb2.Assertion.WithinBounds(
              lower_bound=100,
              upper_bound=200,
            ),
            inclusive_bounds=False,
          )
        }
      )
    )
    event = metrics_pb2.TestCompletedEvent(
      benchmark_id="test_benchmark",
      duration=duration_pb2.Duration(seconds=150),
      metric_collection_config=metrics_pb2.MetricCollectionConfig(
        sources=[metric_source]
      )
    )
    collector = literal_collector.LiteralCollector(event=event, raw_source=metric_source)
    points = collector.metric_points()
    self.assertLen(points, 1)
    self.assertEqual(points[0].metric_key, 'duration')
    self.assertEqual(points[0].metric_value, 150)
    self.assertSequenceEqual(
        points[0].bounds, base.Bounds(100, 200, False))

if __name__ == '__main__':
  absltest.main()
