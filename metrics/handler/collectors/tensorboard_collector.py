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

import collections
import itertools
import pathlib

from absl import logging
import numpy as np
from tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf

from handler.collectors import base
import metrics_pb2

# Represents a single summary scalar and the time it was recorded.
TensorBoardScalar = collections.namedtuple('TensorBoardScalar', ['metric_value', 'wall_time'])

class TensorBoardCollector(base.BaseCollector):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _prefixed_tag(self, tag: str, run: str) -> str:
    if not self._source.merge_runs and run != '.':
      return '/'.join((run, tag))

    return tag

  def _include_tag(self, tag: str) -> bool:
    """Determines if a tag should be included in the results.
    
    Includes tags if:
    1) There is an exactly-matching assertion.
    2) There is a matching pattern in include_tags, but no matching patterns
       in exclude tags.
    """
    tag_path = pathlib.PurePath(tag)
    return any(tag == a.tag for a in self._source.aggregate_assertions) or (
        any(tag_path.match(x.tag_pattern) for x in self._source.include_tags)
        and not any(tag_path.match(pattern) for pattern in self._source.exclude_tags))

  def _read_metrics_from_events_dir(self) -> dict:
    """Collect the TensorBoard summary values for each metric.

    Returns:
      raw_metrics (dict): Keys are TensorBoard tags and value is a list of
        TensorBoardScalar for every data point for that Tensorboard tag.
    """
    em = event_multiplexer.EventMultiplexer()
    em.AddRunsFromDirectory(self.output_path)
    em.Reload()

    raw_metrics = collections.defaultdict(list)
    for run, tags in em.Runs().items():
      # 'Old-style' runs have a simple format and store values directly.
      for tag in tags['scalars']:
        prefixed_tag = self._prefixed_tag(tag, run)
        if not self._include_tag(prefixed_tag):
          continue

        raw_metrics[prefixed_tag].extend(
            TensorBoardScalar(metric_value=x.value, wall_time=x.wall_time)
            for x in em.Scalars(run, tag))
      # 'New-style' runs stores values inside of Tensor protos.
      for tag in tags['tensors']:
        prefixed_tag = self._prefixed_tag(tag, run)
        if not self._include_tag(prefixed_tag):
          continue

        for t in em.Tensors(run, tag):
          tensor_dtype = tf.dtypes.as_dtype(t.tensor_proto.dtype)
          try:
            val = np.frombuffer(
                t.tensor_proto.tensor_content,
                tensor_dtype.as_numpy_dtype).tolist()
            assert len(val) == 1  # There should be 1 value per tensor.
            raw_metrics[prefixed_tag].append(
                TensorBoardScalar(metric_value=val[0], wall_time=t.wall_time))
          except ValueError as e:
            logging.warning(
                'Unable to parse tag: `{}` from tensor_content: {}. '
                'Error: {}. Consider adding this tag to exclude_tags '
                'in config.'.format(tag, t.tensor_proto.tensor_content, e))
    
    return raw_metrics

  def read_metrics_and_assertions(self):
    """Yields metric keys, aggregated metric values, and assertions.

    Metric keys are formatted as {tb_tag_name}_{aggregation_strategy}

    See metrics.proto for available aggregation strategies.
    """
    raw_metrics = self._read_metrics_from_events_dir()
    if not raw_metrics:
      logging.warning('`raw_metrics` is empty, skipping metric aggregation.')
      return

    def _aggregate(scalars, strategy):
      if strategy == metrics_pb2.TensorBoardSource.FINAL:
        # Take the MetricPoint with the latest wall_time.
        final_metric = max(scalars, key=lambda p: p.wall_time)
        return final_metric.metric_value
      elif strategy == metrics_pb2.TensorBoardSource.MAX:
        return max(m.metric_value for m in scalars)
      elif strategy == metrics_pb2.TensorBoardSource.MIN:
        return min(m.metric_value for m in scalars)
      elif strategy == metrics_pb2.TensorBoardSource.AVERAGE:
        return np.mean([m.metric_value for m in scalars])
      elif strategy == metrics_pb2.TensorBoardSource.MEDIAN:
        return np.median([m.metric_value for m in scalars])
      else:
        raise NotImplementedError('Unknown aggregation strategy: {}'.format(
            metrics_pb2.TensorBoardSource.AggregationStrategy.Name(strategy)))

    tag_to_aggregate_assertions = collections.defaultdict(list)
    for aggregate_assertion in self._source.aggregate_assertions:
      tag_to_aggregate_assertions[aggregate_assertion.tag].append(
          aggregate_assertion)

    for tag, scalars in raw_metrics.items():
      tag_path = pathlib.PurePath(tag)
      nested_strategies = (x.strategies for x in self._source.include_tags
                           if tag_path.match(x.tag_pattern) and not 
                           any(tag_path.match(pattern) for pattern in self._source.exclude_tags))
      strategies = set(itertools.chain(*nested_strategies))

      if tag in tag_to_aggregate_assertions:
        strategy_to_assertion = {
          aggregate_assertion.strategy: aggregate_assertion.assertion
          for aggregate_assertion in tag_to_aggregate_assertions[tag]
        }
        strategies.update(strategy_to_assertion.keys())
      else:
        strategy_to_assertion = {}

      for strategy in strategies:
        assertion = strategy_to_assertion.get(strategy)
        strategy_name = metrics_pb2.TensorBoardSource.AggregationStrategy.Name(
            strategy).lower()
        metric_key = '{}_{}'.format(tag, strategy_name)
        yield metric_key, _aggregate(scalars, strategy), assertion
