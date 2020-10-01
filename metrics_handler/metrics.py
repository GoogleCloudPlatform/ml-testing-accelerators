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
import time
import traceback

from absl import logging
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import enums
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_multiplexer


ALLOWED_COMPARISONS = ['greater', 'less', 'equal', 'less_or_equal',
                       'greater_or_equal']
_METRIC_CLIENT = monitoring_v3.MetricServiceClient()


MetricPoint = collections.namedtuple('MetricPoint', ['metric_value', 'wall_time'])
Threshold = collections.namedtuple('Threshold', ['type', 'value'])


def compute_memory_metrics(project, job_name):
  """Compute the memory used by a Kubernetes job.

  Args:
    project (string): GCP project name.
    job_name (string): Kubernetes job name.

  Returns:
    memory_metrics (dict): Keys are strings and value for each key is a
      MetricPoint.
  """
  memory_metrics = {}
  lookup_path = _METRIC_CLIENT.project_path(project)
  base_filter = """metric.type = "kubernetes.io/container/{}" AND resource.labels.pod_name = starts_with("{}")"""
  vm_mem_filter = base_filter.format('memory/used_bytes', job_name)
  gpu_mem_filter = base_filter.format('accelerator/memory_used', job_name)
  end_time = time.time()
  interval = monitoring_v3.types.TimeInterval()
  interval.end_time.seconds = int(end_time)
  interval.end_time.nanos = int((end_time - interval.end_time.seconds) * 10**9)
  interval.start_time.seconds = int(end_time - 60*60*48)
  interval.start_time.nanos = 0
  view = enums.ListTimeSeriesRequest.TimeSeriesView.FULL
  for tup in [(vm_mem_filter, 'vm_memory_usage_bytes'),
              (gpu_mem_filter, 'gpu_memory_usage_bytes')]:
    max_value = 0
    try:
      for series in _METRIC_CLIENT.list_time_series(
          lookup_path, tup[0], interval, view):
        max_value = max(
            [max_value] + [p.value.int64_value for p in series.points])
      if max_value:
        memory_metrics[tup[1]] = MetricPoint(max_value, end_time)
    except Exception as e:
      logging.error('Encountered exception when searching for metric {}. '
                    'Exception was: '.format(tup[0], traceback.format_exc()))
  return memory_metrics

def get_computed_metrics(raw_metrics_dict, job_status_dict,
                         project_id=None, job_name=None, tta_config=None,
                         find_memory_metrics=True):
  """Computes additional metrics about a test.

  Args:
    raw_metrics_dict (dict): Keys are strings and values are MetricPoints.
    job_status_dict (dict): Should contain `job_status`, `start_time`,
      and `stop_time` as keys.
    project_id (string, optional): Name of the GCP project where the test ran.
      Required if `find_memory_metrics` is True.
    job_name (string, optional): Name of the Kubernetes job for this run of the
      test. Required if `find_memory_metrics` is True.
    tta_config (dict, optional): The `time_to_accuracy` portion of the
      metric collection config. Should contain `accuracy_tag` and
      `accuracy_threshold` as keys. If absent, `time_to_accuracy` will not
      be added to the return value of this method.
    find_memory_metrics (bool, optional): If True, query Cloud Monitoring
      to find memory usage metrics and add them to `metrics_dict`.

  Returns:
    computed_metrics_dict (dict): Keys are strings and values are
      MetricPoints.
  """
  computed_metrics_dict = {}
  if 'TensorboardStartTimestamp' in raw_metrics_dict:
    custom_start_time = raw_metrics_dict[
        'TensorboardStartTimestamp'][-1].metric_value
    logging.warning('Using TensorboardStartTimestamp for start time with '
                    'value: {}'.format(custom_start_time))
    start_time = custom_start_time
  else:
    start_time = job_status_dict['start_time']
  stop_time = job_status_dict['stop_time']
  computed_metrics_dict['total_wall_time'] = MetricPoint(
      stop_time - start_time, stop_time)

  # Compute time_to_accuracy if requested in the config.
  if tta_config:
    if 'accuracy_tag' not in tta_config or \
        'accuracy_threshold' not in tta_config:
      raise ValueError('Invalid `time_to_accuracy` portion of config. '
                       'See README for how to set up the config.')
    tag = tta_config['accuracy_tag']
    threshold = tta_config['accuracy_threshold']
    computed_metrics_dict['time_to_accuracy'] = time_to_accuracy(
        raw_metrics_dict, tag, threshold, start_time)

  if find_memory_metrics:
    if not project_id or not job_name:
      raise ValueError('project_id and job_name are required if '
                       'find_memory_metrics=True')
    computed_metrics_dict.update(
        compute_memory_metrics(project_id, job_name))
  return computed_metrics_dict

def read_metrics_from_events_dir(events_dir, tags_to_ignore=None,
                                 use_run_name_prefix=False):
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
  tags_to_ignore = tags_to_ignore or set()

  em = event_multiplexer.EventMultiplexer()
  em.AddRunsFromDirectory(events_dir)
  em.Reload()

  raw_metrics = collections.defaultdict(list)
  for run, tags in em.Runs().items():
    # 'Old-style' runs have a simple format and store values directly.
    for tag in tags['scalars']:
      if use_run_name_prefix and run != '.':
        tag_final = '/'.join((run, tag))
      else:
        tag_final = tag
      if tag_final in tags_to_ignore:
        continue
      raw_metrics[tag_final].extend(
          [MetricPoint(metric_value=x.value, wall_time=x.wall_time)
          for x in em.Scalars(run, tag)])
    # 'New-style' runs stores values inside of Tensor protos.
    for tag in tags['tensors']:
      if use_run_name_prefix and run != '.':
        tag_final = '/'.join((run, tag))
      else:
        tag_final = tag
      if tag_final in tags_to_ignore:
        continue
      for t in em.Tensors(run, tag):
        tensor_dtype = tf.dtypes.as_dtype(t.tensor_proto.dtype)
        try:
          val = np.frombuffer(
              t.tensor_proto.tensor_content,
              tensor_dtype.as_numpy_dtype).tolist()
          assert len(val) == 1  # There should be 1 value per tensor.
          raw_metrics[tag_final].append(
              MetricPoint(metric_value=val[0], wall_time=t.wall_time))
        except ValueError as e:
          logging.warning(
              'Unable to parse tag: `{}` from tensor_content: {}. '
              'Error: {}. Consider adding this tag to tags_to_ignore '
              'in config.'.format(tag, t.tensor_proto.tensor_content, e))

  return raw_metrics


def aggregate_metrics(raw_metrics, default_strategies, metric_strategies=None):
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
  if not raw_metrics:
    logging.warning('`raw_metrics` is empty, skipping metric aggregation.')
    return {}
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
    elif strategy == 'average':
      average_metric_value = np.mean([m.metric_value for m in metrics])
      average_metric_time = np.mean([m.wall_time for m in metrics])
      return MetricPoint(average_metric_value, average_metric_time)
    else:
      raise ValueError('Unknown aggregation strategy: {}'.format(strategy))

  final_metrics = {}
  for tag, metrics in raw_metrics.items():
    strategies = metric_strategies.get(tag, default_strategies)
    for strategy in strategies:
      metric_name = '{}_{}'.format(tag, strategy)
      final_metrics[metric_name] = _aggregate(metrics, strategy)

  return final_metrics


def time_to_accuracy(raw_metrics, tag, threshold, start_wall_time):
  """Calculate the amount of time for accuracy to cross a given threshold.

  Args:
    raw_metrics (dict): Mapping of TensorBoard tags to list of MetricPoint.
    tag (string): Name of accuracy metric. Should be a key in `raw_metrics`.
    threshold (float): The desired model accuracy.
    start_wall_time (float): Wall time at which the test began.

  Returns:
    float, amount of time in seconds to reach the desired accuracy.
  """
  values = raw_metrics.get(tag)
  if not values:
    raise ValueError('No values found for time to accuracy tag: {}. '
        'Possible tags were: {}'.format(tag, raw_metrics.keys()))

  # MetricPoints should be sorted by timestamp with earlier events first.
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
    threshold (Threshold): Desired metric threshold.
    comparison (string): Comparison to given threshold.

  Returns:
    lower_bound (float): Lower boundary that this metric should be above.
    upper_bound (float): Upper boundary that this metric should be below.

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
    if comparison not in (
        'greater', 'greater_or_equal', 'less', 'less_or_equal'):
      raise ValueError(
          'A metric success condition using a `stddevs_from_mean`-type '
          'threshold must use `greater`, `greater_or_equal`, `less`, or '
          '`less_or_equal` for the comparison. The comparison '
          'was: {}'.format(comparison))
    values = [v.metric_value for v in value_history]
    mean = np.mean(values)
    stddev = np.std(values)

    if 'less' in comparison:
      lower_bound = -math.inf
      upper_bound = mean + (stddev * threshold.value)
    elif 'greater' in comparison:
      lower_bound = mean - (stddev * threshold.value)
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
