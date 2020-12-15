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

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import duration_pb2

import utils
from . import base
from . import literal_collector

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
    self.assertEqual(
        points[0].bounds, utils.Bounds(100, 200, False))

if __name__ == '__main__':
  absltest.main()
