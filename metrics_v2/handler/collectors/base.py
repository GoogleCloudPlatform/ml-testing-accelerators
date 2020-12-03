import abc
import collections
import math
import typing

import numpy as np

from handler import utils
import metrics_pb2


def _get_historical_data_for_metric(benchmark_id, metric_key, time_window=None, start_time=None):
  raise NotImplementedError

class BaseCollector:
  """
  Base class for Collector implementations.
  """
  def __init__(self, event, raw_source):
    self._event = event
    if raw_source:
      self._source = getattr(raw_source, raw_source.WhichOneof('source_type'))

  @property
  def benchmark_id(self):
    return (self._event.metric_collection_config.compare_to_benchmark_id 
        or self._event.benchmark_id)

  @property
  def output_path(self):
    return self._event.output_path

  @property
  def source(self):
    return self._source

  def read_metrics_and_assertions(self):
    raise NotImplementedError

  def compute_bounds(self, name, assertion) -> utils.Bounds:
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
      values = _get_historical_data_for_metric(
        self.benchmark_id,
        name,
        assertion.time_window.ToTimedelta(),
        assertion.start_time.ToDatetime())
      mean = np.mean(values)
      stddev = np.std(values)

      c = assertion.std_devs_from_mean.comparison
      if c in (metrics_pb2.Assertion.LESS, metrics_pb2.Assertion.WITHIN):
        upper_bound = mean + (stddev * assertion.std_devs_from_mean.std_devs)
      if c in (metrics_pb2.Assertion.GREATER, metrics_pb2.Assertion.WITHIN):
        lower_bound = mean - (stddev * assertion.std_devs_from_mean.std_devs)
    elif assertion_type == 'percent_difference':
      target_type = assertion.percent_difference.WhichOneof('target_type')
      if target_type == 'use_historical_mean':
        values = _get_historical_data_for_metric(
          self.benchmark_id,
          name,
          assertion.time_window.ToTimedelta(),
          assertion.start_time.ToDatetime())
        target = np.mean(values)
      elif target_type == 'value':
        target = assertion.percent_difference.target

      c = assertion.percent_difference.comparison
      if c in (metrics_pb2.Assertion.LESS, metrics_pb2.Assertion.WITHIN):
        upper_bound = target + (assertion.percent_difference.percent * target)
      if c in (metrics_pb2.Assertion.GREATER, metrics_pb2.Assertion.WITHIN):
        lower_bound = target - (assertion.percent_difference.percent * target)
    
    if upper_bound == math.inf and lower_bound == math.inf:
      logging.error(
          '%s: comparison %s is not implemented for assertion type `%s`',
          name, metrics_pb2.Assertion.Comparison.Name(c), assertion_type)
      return utils.NO_BOUNDS

    return utils.Bounds(lower_bound, upper_bound, inclusive)

  def metric_points(self) -> typing.Iterable[utils.MetricPoint]:
    return [utils.MetricPoint(key, value, self.compute_bounds(key, assertion)) for key, value, assertion in self.read_metrics_and_assertions()]
