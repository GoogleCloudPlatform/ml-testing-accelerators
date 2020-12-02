import json
import os

from handler.collectors import base

from absl import logging
import tensorflow as tf

class PerfZeroCollector(base.BaseCollector):
  def read_metrics_and_assertions(self):
    """Retrieves aggregated metrics from a PerfZero summary file.

    Returns:
      aggregated_metrics (dict):  Key is metric name and value is a MetricPoint
        containing the aggregated value for that metric.
    """
    glob_pattern = os.path.join(self.output_path, '*', 'perfzero_summary.json')
    file_matches = tf.io.gfile.glob(glob_pattern)
    if not file_matches:
      logging.info('No perfzero summary found.')
      return

    file_path = file_matches[0]
    with tf.io.gfile.GFile(file_path) as f:
      summary = json.loads(f.read())
      timestamp = summary["execution_timestamp"]
      for pair in summary["benchmark_result"]["metrics"]:
        key, value = pair["name"], pair["value"]
        yield key, value, self._source.assertions.get(key)

      # Process info metrics are recorded as "process_info/..."
      for key, value in summary["process_info"].items():
        prefixed_key = os.path.join("process_info", key)
        yield prefixed_key, value, self._source.assertions.get(prefixed_key)


      yield  "total_wall_time", summary["benchmark_result"]["wall_time"], self._source.assertions.get("total_wall_time")
