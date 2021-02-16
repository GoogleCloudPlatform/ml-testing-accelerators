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

from google.protobuf import duration_pb2

from handler.collectors import base

class LiteralCollector(base.BaseCollector):
  def read_metrics_and_assertions(self):
    for metric_key, assertion in self._source.assertions.items():
      try:
        # TODO: implement lookups in nested fields
        raw_value = getattr(self._event, metric_key)
      except AttributeError:
        logging.error('Attribute %s not found in event', metric_key)
        continue
      
      try:
        if type(raw_value) is duration_pb2.Duration:
          value = raw_value.ToTimedelta().total_seconds()
        else:
          value = float(raw_value)
      except TypeError:
        logging.error('%s=%s could not be parsed as float', metric_key, str(raw_value))

      yield metric_key, value, assertion
