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

  def _prefixed_tag(self, tag):
    if self.source.merge_runs and run != '.':
      return '/'.join((run, tag))

    return tag


  def _include_tag(self, tag):
    tag_path = pathlib.PurePath(tag)
    return tag in self.source.assertions or (
        any(tag_path.match(x.tag_pattern) for x in self.source.include_tags)
        and not any(tag_path.match(pattern) for pattern in self.source.exclude_tags))

  def _read_metrics_from_events_dir(self):
    """Collect the TensorBoard summary values for each metric.

    Args:
      events_dir (string): Path to location of TensorBoard summaries.
      tags_to_ignore (set[string]): Set of TensorBoard tag names to skip.
      use_run_name_prefix (bool): If True, prefix tag names with the name
        of the run that contains them (e.g. `train/` or `eval/`)

    Returns:
      raw_metrics (dict): Keys are TensorBoard tags and value is a list of
        MetricPoint for every data point for that Tensorboard tag.
    """
    em = event_multiplexer.EventMultiplexer()
    em.AddRunsFromDirectory(self.output_path)
    em.Reload()

    raw_metrics = collections.defaultdict(list)
    for run, tags in em.Runs().items():
      # 'Old-style' runs have a simple format and store values directly.
      for tag in tags['scalars']:
        prefixed_tag = self._prefixed_tag(tag)
        if not self._include_tag(prefixed_tag):
          continue

        raw_metrics[prefixed_tag].extend(
            TensorBoardScalar(metric_value=x.value, wall_time=x.wall_time)
            for x in em.Scalars(run, tag))
      # 'New-style' runs stores values inside of Tensor protos.
      for tag in tags['tensors']:
        prefixed_tag = self._prefixed_tag(tag)
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
                'Error: {}. Consider adding this tag to tags_to_ignore '
                'in config.'.format(tag, t.tensor_proto.tensor_content, e))
    
    return raw_metrics

  def read_metrics_and_assertions(self):
    """Aggregate raw TensorBoard metrics according to collection config.

    Available aggregation strategies: `final`, `min`, `max`, `average`.

    Args:
      raw_metrics: dict mapping TensorBoard tags to list of MetricPoint.
      default_strategies: list of aggregation strategies to use as the default.
      metric_strategies: dict mapping tags to aggregation strategies.

    Returns:
      dict mapping metric name to MetricPoint.

    Raises:
      ValueError: If `default_strategies` is empty or an invalid strategy is
        provided.
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

    final_metrics = {}
    for tag, scalars in raw_metrics.items():
      tag_path = pathlib.PurePath(tag)
      nested_strategies = (x.strategies for x in self.source.include_tags
                           if tag_path.match(x.tag_pattern) and not 
                           any(tag_path.match(pattern) for pattern in self.source.exclude_tags))
      strategies = set(itertools.chain(*nested_strategies))

      if tag in self.source.assertions:
        strategy_to_assertion = {
            a.strategy: a.assertion
            for a in self.source.assertions[tag].aggregate_assertions
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
