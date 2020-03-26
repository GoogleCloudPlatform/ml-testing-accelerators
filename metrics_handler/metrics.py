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
import math

from absl import logging
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_multiplexer


ALLOWED_COMPARISONS = ['greater', 'less', 'equal', 'less_or_equal']


MetricPoint = collections.namedtuple('MetricPoint', ['metric_value', 'wall_time'])
Threshold = collections.namedtuple('Threshold', ['type', 'value'])


def read_metrics_from_events_dir(events_dir, tags_to_ignore=None):
  """Collect the TensorBoard summary values for each metric.
  
  Args:
    events_dir: path to location of TensorBoard summaries.
    tags_to_ignore: set of TensorBoard tag names to skip.

  Returns:
    dict mapping TensorBoard tags to list of MetricPoint.
  """
  tags_to_ignore = tags_to_ignore or set()

  em = event_multiplexer.EventMultiplexer()
  em.AddRunsFromDirectory(events_dir)
  em.Reload()

  raw_metrics = collections.defaultdict(list)
  for run, tags in em.Runs().items():
    # 'Old-style' runs have a simple format and store values directly.
    for tag in tags['scalars']:
      if tag in tags_to_ignore:
        continue
      raw_metrics[tag].extend(
          [MetricPoint(metric_value=x.value, wall_time=x.wall_time)
          for x in em.Scalars(run, tag)])
    # 'New-style' runs stores values inside of Tensor protos.
    for tag in tags['tensors']:
      if tag in tags_to_ignore:
        continue
      for t in em.Tensors(run, tag):
        tensor_dtype = tf.dtypes.as_dtype(t.tensor_proto.dtype)
        try:
          val = np.frombuffer(
              t.tensor_proto.tensor_content,
              tensor_dtype.as_numpy_dtype).tolist()
          assert len(val) == 1  # There should be 1 value per tensor.
          raw_metrics[tag].append(
              MetricPoint(metric_value=val[0], wall_time=t.wall_time))
        except ValueError as e:
          logging.warning(
              'Unable to parse tag: `{}` from tensor_content: {}. '
              'Error: {}. Consider adding this tag to tags_to_ignore '
              'in config.'.format(tag, t.tensor_proto.tensor_content, e))

  return raw_metrics


def aggregate_metrics(raw_metrics, default_strategies, metric_strategies=None):
  """Aggregate raw TensorBoard metrics according to collection config.

  Available aggregation strategies: `final`, `min`, `max`.

  Args:
    raw_metrics: dict mapping TensorBoard tags to list of MetricPoint.
    default_strategies: list of aggregation strategies to use as the default.
    metric_strategies: dict mapping tags to aggregation strategies.

  Returns:
    dict mapping metric name to MetricPoint.
  """
  if not raw_metrics:
    raise ValueError('`raw_metrics` cannot be empty.')
  if not default_strategies:
    raise ValueError('`default_strategies` cannot be empty.')
  metric_strategies = metric_strategies or {}

  def _aggregate(metrics, strategy):
    if strategy == 'final':
      # Take the MetricPoint with the latest wall_time.
      latest_metric = metrics[0]
      for metric in metrics[1:]:
        if metric.wall_time > latest_metric.wall_time:
          latest_metric = metric
      return latest_metric
    elif strategy == 'max':
      # Take the MetricPoint with the maximum metric value.
      max_metric = metrics[0]
      for metric in metrics[1:]:
        if metric.metric_value > max_metric.metric_value:
          max_metric = metric
      return max_metric
    elif strategy == 'min':
      # Take the MetricPoint with the minimum metric value.
      min_metric = metrics[0]
      for metric in metrics[1:]:
        if metric.metric_value < min_metric.metric_value:
          min_metric = metric
      return min_metric
    else:
      raise ValueError('Unknown aggregation strategy: {}'.format(strategy))

  final_metrics = {}
  for tag, metrics in raw_metrics.items():
    strategies = metric_strategies.get(tag, default_strategies)
    for strategy in strategies:
      metric_name = '{}_{}'.format(tag, strategy)
      final_metrics[metric_name] = _aggregate(metrics, strategy)

  return final_metrics


def total_wall_time(raw_metrics):
  """Calculate the total wall time from TensorBoard summaries.
  
  Args:
    raw_metrics: dict mapping TensorBoard tags to list of MetricPoint.

  Returns:
    float, difference in wall time between first and last summaries.
  """
  values = list(itertools.chain.from_iterable(raw_metrics.values()))
  min_wall_time = min(v.wall_time for v in values)
  max_wall_time = max(v.wall_time for v in values)

  return MetricPoint(max_wall_time - min_wall_time, max_wall_time)


def time_to_accuracy(raw_metrics, tag, threshold):
  """Calculate the amount of time for accuracy to cross a given threshold.

  Args:
    raw_metrics: dict mapping TensorBoard tags to list of MetricPoint.
    tag: string name of accuracy metric.
    threshold: the desired model accuracy.
  
  Returns:
    float, amount of time in seconds to reach the desired accuracy.
  """
  values = raw_metrics.get(tag)
  if not values:
    raise ValueError('No values found for time to accuracy tag: {}. '
        'Possible tags were: {}'.format(tag, raw_metrics.keys()))

  # MetricPoints should be sorted by timestamp with earlier events first.
  start_wall_time = values[0].wall_time
  try:
    end_wall_time = next(
        v.wall_time for v in values
        if v.metric_value >= threshold)
    return MetricPoint(end_wall_time - start_wall_time, end_wall_time)
  except StopIteration:
    max_accuracy = max(v.metric_value for v in values)
    raise ValueError(
        'Accuracy metric `{}` was never high enough to satisfy the '
        '`time_to_accuracy` settings from the config. Max accuracy: {}. '
        'Target accuracy: {}. Config for `time_to_accuracy`: {}'.format(
            tag, max_accuracy, threshold))


def metric_bounds(value_history, threshold, comparison):
  """Compute upper/lower bounds and whether metric is within those bounds.

  Args:
    value_history (list of floats): History of values for this metric. These
      should be ordered so the most recent value is the last in the list.
    threshold: Threshold, desired metric threshold.
    comparison: string, comparison to given threshold.

  Returns:
    tuple(is_within_bounds (bool), lower_bound (float), upper_bound (float)).
      lower_bound and/or upper_bound can be None.

  Raises:
    ValueError if the regression test config is invalid.
  """
  if not comparison or comparison not in ALLOWED_COMPARISONS:
    raise ValueError(
        'A metric success condition must set the `comparison` field to '
        'one of {}. comparison was: {}'.format(
            ALLOWED_COMPARISONS, comparison))

  if threshold.type == 'fixed_value':
    if 'greater' in comparison:
      lower_bound = threshold.value
      upper_bound = math.inf
    elif 'less' in comparison:
      lower_bound = -math.inf
      upper_bound = threshold.value
    elif comparison == 'equal':
      lower_bound = threshold.value
      upper_bound = threshold.value
    else:
      raise ValueError(
          'A metric success condition using a `fixed_value`-type threshold '
          'must use `greater`, `greater_or_equal`, `less`, `less_or_equal`, '
          'or `equal` for the comparison. The comparison was: {}'.format(
              comparison))
  elif threshold.type == 'stddevs_from_mean':
    if comparison not in ('greater', 'less', 'less_or_equal'):
      raise ValueError(
          'A metric success condition using a `stddevs_from_mean`-type '
          'threshold must use `greater`, `less`, or `less_or_equal` for the '
          'comparison. The comparison was: {}'.format(comparison))
    values = [v.metric_value for v in value_history]
    mean = np.mean(values)
    stddev = np.std(values)

    if 'less' in comparison:
      lower_bound = -math.inf
      upper_bound = mean + (stddev * threshold.value)
    elif 'greater' in comparison:
      lower_bound = max(mean - (stddev * threshold.value), 0)
      upper_bound = math.inf
  else:
    raise ValueError(
      'The threshold type of a metric success condition should be either '
      '`fixed_value` or `stddevs_from_mean`. Condition was: {}'.format(
          success_condition))

  return lower_bound, upper_bound


def within_bounds(value, lower_bound, upper_bound, inclusive=False):
  """Determine whether given value is within bounds.

  Args:
    value: float, value to test.
    lower_bound: float, minimum acceptable value.
    upper_bound: float, maximum acceptable value.
    inclusive: optional float, whether to check if metric is close to bounds.

  Returns:
    boolean, whether metric is within the given bounds.
  """
  if inclusive and (
      math.isclose(value, lower_bound) or math.isclose(value, upper_bound)):
    return True
  
  return value > lower_bound and value < upper_bound
