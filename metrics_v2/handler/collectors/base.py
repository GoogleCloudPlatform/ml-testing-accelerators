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
import math
import typing

from absl import logging
import numpy as np
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2

from handler import utils
import metrics_pb2


class BaseCollector:
  """
  Base class for Collector implementations.
  """
  def __init__(self, event, raw_source: metrics_pb2.MetricSource, metric_store=None):
    self._event = event
    if raw_source:
      self._source = getattr(raw_source, raw_source.WhichOneof('source_type'))
    self._metric_store =  metric_store

  @property
  def output_path(self):
    """Output path from test event."""
    return self._event.output_path

  def read_metrics_and_assertions(self) -> (
      typing.Iterable[typing.Tuple[str, float, metrics_pb2.Assertion]]):
    """Yields unique metric keys, values, and the corresponding assertions."""
    raise NotImplementedError

  def get_metric_history(
      self,
      metric_key: str,
      time_window: duration_pb2.Duration,
      min_timestamp: timestamp_pb2.Timestamp
  ) -> typing.List[float]:
    """Returns the historical values of a given metric.
    
    Args:
      metric_key: Unique string identifying the name of the metric.
      time_window: Time limit for metric history, relative to the test start
        time. Data points older than the time window will not be returned.
      min_timestamp: Absolute time limit for metric history. Data points older
        than this timestamp will not be returned.

    Returns:
      A list of the historical values of the given metric as floats.
    """
    if not self._metric_store:
      raise ValueError('Metric history requested for {}, but no metric store '
                       'was provided to Collector.'.format(metric_key))

    if time_window.ToTimedelta():
      min_time = max(
          min_timestamp.ToDatetime(),
          self._event.start_time.ToDatetime() - time_window.ToTimedelta())
    else:
      min_time = min_timestamp.ToDatetime()

    history_rows = self._metric_store.get_metric_history(
      benchmark_id=(
          self._event.metric_collection_config.compare_to_benchmark_id or
          self._event.benchmark_id),
      metric_key=metric_key,
      min_time=min_time,
    )
    
    return [row.metric_value for row in history_rows]

  def compute_bounds(
      self,
      metric_key: str,
      assertion: metrics_pb2.Assertion
  ) -> utils.Bounds:
    """Returns the bounds for a given metric, based on the given assertion.

    This method may result in database calls to gather historical data for some
    types of assertions.

    Args:
      metric_key: Unique string identifying the name of the metric.
      assertion: The assertion that will be used to define the bounds.

    Returns:
      An instance of utils.Bounds representing the metric bounds.
    """
    if assertion is None:
      return utils.NO_BOUNDS

    lower_bound = -math.inf
    upper_bound = math.inf
    inclusive = assertion.inclusive_bounds

    assertion_type = assertion.WhichOneof('assertion_type')
    if assertion_type == 'fixed_value':
      c = assertion.fixed_value.comparison
      if c == metrics_pb2.Assertion.LESS:
        upper_bound = assertion.fixed_value.value
      elif c == metrics_pb2.Assertion.GREATER:
        lower_bound = assertion.fixed_value.value
      elif c == metrics_pb2.Assertion.EQUAL:
        lower_bound = assertion.fixed_value.value
        upper_bound = assertion.fixed_value.value
        inclusive = True
    elif assertion_type == 'within_bounds':
      lower_bound = assertion.within_bounds.lower_bound
      upper_bound = assertion.within_bounds.upper_bound
    elif assertion_type == 'std_devs_from_mean':
      values = self.get_metric_history(
        metric_key,
        assertion.time_window,
        assertion.min_timestamp)
      
      # Standard deviation not defined for n < 2
      min_num_points = max(assertion.wait_for_n_data_points, 2)
      if len(values) < min_num_points:
        logging.info('Not enough data points to compute bounds for %s. '
                     'Need %d points, have %d.',
                     metric_key, min_num_points, len(values))
        return utils.NO_BOUNDS

      mean = np.mean(values)
      stddev = np.std(values)
      c = assertion.std_devs_from_mean.comparison
      if c in (metrics_pb2.Assertion.LESS, metrics_pb2.Assertion.WITHIN):
        upper_bound = mean + (stddev * assertion.std_devs_from_mean.std_devs)
      if c in (metrics_pb2.Assertion.GREATER, metrics_pb2.Assertion.WITHIN):
        lower_bound = mean - (stddev * assertion.std_devs_from_mean.std_devs)

      if upper_bound == math.inf and lower_bound == -math.inf:
        logging.error(
            '%s: comparison %s is not implemented for assertion type `%s`',
            metric_key, metrics_pb2.Assertion.Comparison.Name(c), assertion_type)
        return utils.NO_BOUNDS
    elif assertion_type == 'percent_difference':
      target_type = assertion.percent_difference.WhichOneof('target_type')
      if target_type == 'use_historical_mean':
        values = self.get_metric_history(
          metric_key,
          assertion.time_window,
          assertion.min_timestamp)

        # Mean not defined for n < 1.
        min_num_points = max(assertion.wait_for_n_data_points, 1)
        if len(values) < min_num_points:
          logging.info('Not enough data points to compute bounds for %s. '
                       'Need %d points, have %d.',
                       metric_key, len(values), min_num_points)
          return utils.NO_BOUNDS
        target = np.mean(values)
      elif target_type == 'value':
        target = assertion.percent_difference.value
      else:
        logging.error('%s: No `target_type` defined for assertion type `%s`.',
                      metric_key, assertion_type)
        return utils.NO_BOUNDS

      c = assertion.percent_difference.comparison
      if c in (metrics_pb2.Assertion.LESS, metrics_pb2.Assertion.WITHIN):
        upper_bound = target + (assertion.percent_difference.percent * target)
      if c in (metrics_pb2.Assertion.GREATER, metrics_pb2.Assertion.WITHIN):
        lower_bound = target - (assertion.percent_difference.percent * target)

      if upper_bound == math.inf and lower_bound == -math.inf:
        logging.error(
            '%s: comparison %s is not implemented for assertion type `%s`',
            metric_key, metrics_pb2.Assertion.Comparison.Name(c), assertion_type)
        return utils.NO_BOUNDS

    return utils.Bounds(lower_bound, upper_bound, inclusive)

  def metric_points(self) -> typing.Iterable[utils.MetricPoint]:
    """Returns a list of metric points collected from the test output."""
    return [
        utils.MetricPoint(key, value, self.compute_bounds(key, assertion))
        for key, value, assertion in self.read_metrics_and_assertions()]
