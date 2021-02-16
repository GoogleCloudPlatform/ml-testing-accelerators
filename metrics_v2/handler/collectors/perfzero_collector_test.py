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

import json
import os
import pathlib

from absl.testing import absltest

import base
from handler import utils
from handler.collectors import perfzero_collector
import metrics_pb2

class PerfZeroCollectorTest(absltest.TestCase):
  def test_get_metrics_from_perfzero_summary(self):
    temp_dir = self.create_tempdir().full_path
    summary_dir = os.path.join(temp_dir, 'date_and_time')
    pathlib.Path(summary_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'perfzero_summary.json')
    with open(summary_path, 'w') as f:
      json.dump({
        "execution_id": "execution_id",
        "execution_timestamp": 1234567890.1,
        "benchmark_result": {
          "wall_time": 1234,
          "metrics": [
            {
              "name": "exp_per_second",
              "value": 1.1,
            },
            {
              "name": "avg_exp_per_second",
              "value": 2.2,
            },
            {
              "name": "startup_time",
              "value": 3.3
            }
          ],
        },
        "benchmark_info": {
          "not": "important",
        },
        "setup_info": {},
        "ml_framework_info": {
          "not": "important",
        },
        "system_info": {
          "not": "important"
        },
        "process_info": {
          "max_rss": 4.4,
          "max_vms": 5.5,
          "max_cpu_percent": 6.6,
        }
        }, f)

    metric_source = metrics_pb2.MetricSource(
      perfzero=metrics_pb2.PerfZeroSource(
        assertions={
          'total_wall_time': metrics_pb2.Assertion(
            within_bounds=metrics_pb2.Assertion.WithinBounds(
              lower_bound=1230,
              upper_bound=1240,
            )
          ),
          'exp_per_second': metrics_pb2.Assertion(
            within_bounds=metrics_pb2.Assertion.WithinBounds(
              lower_bound=1,
              upper_bound=100,
            ),
          )
        }
      )
    )
    event = metrics_pb2.TestCompletedEvent(
      benchmark_id="test_benchmark",
      output_path=temp_dir,
      metric_collection_config=metrics_pb2.MetricCollectionConfig(
        sources=[metric_source]
      )
    )

    collector = perfzero_collector.PerfZeroCollector(event=event, raw_source=metric_source)
    points = list(collector.metric_points())
    self.assertCountEqual(
      points,
      {
        utils.MetricPoint("total_wall_time", 1234, utils.Bounds(1230., 1240., False)),
        utils.MetricPoint("exp_per_second", 1.1, utils.Bounds(1., 100., False)),
        utils.MetricPoint("avg_exp_per_second", 2.2, utils.NO_BOUNDS),
        utils.MetricPoint("startup_time", 3.3, utils.NO_BOUNDS),
        utils.MetricPoint("process_info/max_rss", 4.4, utils.NO_BOUNDS),
        utils.MetricPoint("process_info/max_vms", 5.5, utils.NO_BOUNDS),
        utils.MetricPoint("process_info/max_cpu_percent", 6.6, utils.NO_BOUNDS),
      },
    )

if __name__ == '__main__':
  absltest.main()
