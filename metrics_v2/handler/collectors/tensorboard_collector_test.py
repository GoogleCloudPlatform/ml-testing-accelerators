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

import contextlib
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from . import base
from . import tensorboard_collector
import utils
import metrics_pb2

class TensorBoardCollectorTest(parameterized.TestCase):
  def setUp(self):
    self.temp_dir = self.create_tempdir().full_path
    summary_writer = tf.summary.create_file_writer(self.temp_dir)
    with summary_writer.as_default(), contextlib.closing(summary_writer):
      tf.summary.scalar("foo", 1, 0)
      tf.summary.scalar("foo", 2, 100)

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(self.temp_dir, 'train'))
    with train_summary_writer.as_default(), contextlib.closing(train_summary_writer):
      tf.summary.scalar("bar", 10, 0)
      tf.summary.scalar("bar", 100, 200)
      tf.summary.scalar("bar", 100, 100)

    eval_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.temp_dir, 'eval'))
    with eval_summary_writer.as_default(), contextlib.closing(eval_summary_writer):
      tf.summary.scalar("accuracy", .125, 0)
      tf.summary.scalar("accuracy", .5, 100)
      tf.summary.scalar("accuracy", .25, 200)

  def test_aggregate_metrics_include_all_strategies(self):
    metric_source = metrics_pb2.MetricSource(
      tensorboard=metrics_pb2.TensorBoardSource(
        include_tags=[
          metrics_pb2.TensorBoardSource.TagStrategy(
            tag_pattern="*",
            strategies=[
              metrics_pb2.TensorBoardSource.FINAL,
              metrics_pb2.TensorBoardSource.MAX,
              metrics_pb2.TensorBoardSource.MIN,
              metrics_pb2.TensorBoardSource.AVERAGE,
              metrics_pb2.TensorBoardSource.MEDIAN,
            ]
          )
        ]
      )
    )
    event = metrics_pb2.TestCompletedEvent(
      benchmark_id="test_benchmark",
      output_path=self.temp_dir,
      metric_collection_config=metrics_pb2.MetricCollectionConfig(
        sources=[metric_source]
      )
    )
    collector = tensorboard_collector.TensorBoardCollector(event=event, raw_source=metric_source)
    points = list(collector.metric_points())

    metric_to_value = {
      key: value
      for key, value, _ in points
    }

    self.assertDictEqual(
      metric_to_value,
      {
        'foo_final': 2,
        'foo_min': 1,
        'foo_max': 2,
        'foo_average': 1.5,
        'foo_median': 1.5,
        'eval/accuracy_final': .25,
        'eval/accuracy_min': .125,
        'eval/accuracy_max': .5,
        'eval/accuracy_average': np.mean([.125, .25, .5]),
        'eval/accuracy_median': np.median([.125, .25, .5]),
        'train/bar_final': 100,
        'train/bar_min': 10,
        'train/bar_max': 100,
        'train/bar_average': np.mean([10, 100, 100]),
        'train/bar_median': np.median([10, 100, 100]),
      }
    )

    for _, _, bounds in points:
      self.assertEqual(bounds, utils.NO_BOUNDS)


  def test_aggregate_metrics_with_assertion(self):
    metric_source = metrics_pb2.MetricSource(
      tensorboard=metrics_pb2.TensorBoardSource(
        include_tags=[
          metrics_pb2.TensorBoardSource.TagStrategy(
            tag_pattern="eval/*",
            strategies=[
              metrics_pb2.TensorBoardSource.FINAL,
              metrics_pb2.TensorBoardSource.MAX,
              metrics_pb2.TensorBoardSource.MIN,
            ]
          )
        ],
        aggregate_assertions=[
          metrics_pb2.TensorBoardSource.AggregateAssertion(
            tag='eval/accuracy',
            strategy=metrics_pb2.TensorBoardSource.MAX,
            assertion=metrics_pb2.Assertion(
              within_bounds=metrics_pb2.Assertion.WithinBounds(
                lower_bound=.4,
                upper_bound=1.0,
              ),
              inclusive_bounds=True,
            )
          )
        ]
      )
    )
    event = metrics_pb2.TestCompletedEvent(
      benchmark_id="test_benchmark",
      output_path=self.temp_dir,
      metric_collection_config=metrics_pb2.MetricCollectionConfig(
        sources=[metric_source]
      )
    )
    collector = tensorboard_collector.TensorBoardCollector(event=event, raw_source=metric_source)
    points = list(collector.metric_points())

    self.assertCountEqual(
      points,
      [
        utils.MetricPoint('eval/accuracy_max', .5, utils.Bounds(.4, 1.0, True)),
        utils.MetricPoint('eval/accuracy_min', .125, utils.NO_BOUNDS),
        utils.MetricPoint('eval/accuracy_final', .25, utils.NO_BOUNDS),
      ],
    )

  def test_include_and_exclude(self):
    metric_source = metrics_pb2.MetricSource(
      tensorboard=metrics_pb2.TensorBoardSource(
        include_tags=[
          metrics_pb2.TensorBoardSource.TagStrategy(
            tag_pattern="*",
            strategies=[
              metrics_pb2.TensorBoardSource.FINAL,
            ]
          )
        ],
        exclude_tags=[
          'foo',
          'train/*',
        ],
        aggregate_assertions=[
          metrics_pb2.TensorBoardSource.AggregateAssertion(
            tag='foo',
            strategy=metrics_pb2.TensorBoardSource.MIN,
            assertion=metrics_pb2.Assertion(
              within_bounds=metrics_pb2.Assertion.WithinBounds(
                lower_bound=0.,
                upper_bound=2.,
              )
            )
          )
        ]
      )
    )
    event = metrics_pb2.TestCompletedEvent(
      benchmark_id="test_benchmark",
      output_path=self.temp_dir,
      metric_collection_config=metrics_pb2.MetricCollectionConfig(
        sources=[metric_source]
      )
    )
    collector = tensorboard_collector.TensorBoardCollector(event=event, raw_source=metric_source)
    points = list(collector.metric_points())

    self.assertCountEqual(
      points,
      [
        utils.MetricPoint('eval/accuracy_final', .25, utils.NO_BOUNDS),
        utils.MetricPoint('foo_min', 1, utils.Bounds(0., 2., False)),
      ],
    )

if __name__ == '__main__':
  absltest.main()
