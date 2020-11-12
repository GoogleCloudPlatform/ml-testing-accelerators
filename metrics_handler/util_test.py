# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for util."""
import math

from absl.testing import absltest
from absl.testing import parameterized

import util


VALID_LOGS_LINK = 'https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-resnet50-functional-v3-8-1584453600&extra_junk&dateRangeUnbound=backwardInTime'
VALID_DOWNLOAD_COMMAND = """gcloud logging read 'resource.type=k8s_container resource.labels.project_id=xl-ml-test resource.labels.location=us-central1-b resource.labels.cluster_name=xl-ml-test resource.labels.namespace_name=automated resource.labels.pod_name:pt-1.5-resnet50-functional-v3-8-1584453600' --limit 10000000000000 --order asc --format 'value(textPayload)' --project=xl-ml-test > pt-1.5-resnet50-functional-v3-8-1584453600_logs.txt && sed -i '/^$/d' pt-1.5-resnet50-functional-v3-8-1584453600_logs.txt"""
VALID_WORKLOAD_LINK = 'https://console.cloud.google.com/kubernetes/job/us-central1-b/xl-ml-test/automated/pt-1.5-resnet50-functional-v3-8-1584453600?project=xl-ml-test'


class UtilTest(parameterized.TestCase):
  def test_add_unbound_time_to_logs_link_empty_string(self):
    self.assertEqual(
        '',
        util.add_unbound_time_to_logs_link(''))

  def test_add_unbound_time_to_logs_link_already_exists(self):
    self.assertEqual(
        VALID_LOGS_LINK,
        util.add_unbound_time_to_logs_link(VALID_LOGS_LINK))

  def test_add_unbound_time_to_logs_link_success(self):
    self.assertEqual(
        'abc{}'.format(util.UNBOUND_DATE_RANGE),
        util.add_unbound_time_to_logs_link('abc'))

  def test_download_command(self):
    project = 'xl-ml-test'
    cluster = 'xl-ml-test'
    namespace = 'automated'
    pod_name = 'pt-1.5-resnet50-functional-v3-8-1584453600'
    zone = 'us-central1-b'
    self.assertEqual(
        util.download_command(pod_name, namespace, zone, cluster, project),
        VALID_DOWNLOAD_COMMAND)

  def test_workload_link(self):
    project = 'xl-ml-test'
    cluster = 'xl-ml-test'
    namespace = 'automated'
    pod_name = 'pt-1.5-resnet50-functional-v3-8-1584453600'
    zone = 'us-central1-b'
    self.assertEqual(
        util.workload_link(pod_name, namespace, zone, cluster, project),
        VALID_WORKLOAD_LINK)

  @parameterized.named_parameters(
    ('inf', [1, 2, 3, math.inf, 4, 5], [1, 2, 3, None, 4, 5]),
    ('inf_and_str', ['1', 2, 3, math.inf, 4, 5], ['1', 2, 3, None, 4, 5]),
    ('inf_str_none', ['1', None, math.inf, 4, 5], ['1', None, None, 4, 5]),
    ('inf_parsed', [1, 2, 3, float('inf'), 4, 5], [1, 2, 3, None, 4, 5]),
    ('-inf', [1, 2, 3, -math.inf, 4, 5], [1, 2, 3, None, 4, 5]),
    ('nan', [1, 2, 3, math.nan, 4, 5], [1, 2, 3, None, 4, 5]),
    ('nan_parsed', [1, 2, 3, float("nan"), 4, 5], [1, 2, 3, None, 4, 5]),
    ('nan_cap_parsed', [1, 2, 3, float("NAN"), 4, 5], [1, 2, 3, None, 4, 5]),
    ('all', [math.inf, -math.inf, math.nan], [None, None, None]),
    ('one_element', [math.inf], [None]),
    ('empty', [], []),
  )
  def test_replace_invalid_values(self, row, expected_row):
    clean_row = util.replace_invalid_values(row)
    self.assertSequenceEqual(clean_row, expected_row)

  @parameterized.named_parameters(
    ('inf', math.inf, False),
    ('-inf', -math.inf, False),
    ('inf_parsed', float('inf'), False),
    ('-inf_parsed', float('-inf'), False),
    ('nan_parsed', float('nan'), False),
    ('nan_cap_parsed', float('NAN'), False),
    ('int', 9, True),
    ('int_parsed', float('9'), True),
    ('float', 9.0, True),
    ('float_parsed', float('9.0'), True),
  )
  def test_is_valid_bigquery_value(self, value, expected_bool):
    self.assertEqual(util.is_valid_bigquery_value(value), expected_bool)


if __name__ == '__main__':
  absltest.main()
