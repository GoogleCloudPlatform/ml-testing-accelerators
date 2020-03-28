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

import unittest

import util


VALID_LOGS_LINK = 'https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-resnet50-functional-v3-8-1584453600&extra_junk&dateRangeUnbound=backwardInTime'
VALID_DOWNLOAD_COMMAND = """gcloud logging read 'resource.type=k8s_container resource.labels.project_id=xl-ml-test resource.labels.location=us-central1-b resource.labels.cluster_name=xl-ml-test resource.labels.namespace_name=automated resource.labels.pod_name:pt-1.5-resnet50-functional-v3-8-1584453600' --limit 10000000000000 --order asc --format 'value(textPayload)' --project=xl-ml-test > pt-1.5-resnet50-functional-v3-8-1584453600_logs.txt && sed -i '/^$/d' pt-1.5-resnet50-functional-v3-8-1584453600_logs.txt"""
VALID_TEST_NAME = 'pt-1.5-resnet50-functional-v3-8-1584453600'


class UtilTest(unittest.TestCase):
  def test_add_unbound_time_to_logs_link_already_exists(self):
    self.assertEqual(
        VALID_LOGS_LINK,
        util.add_unbound_time_to_logs_link(VALID_LOGS_LINK))

  def test_add_unbound_time_to_logs_link_success(self):
    self.assertEqual(
        'abc{}'.format(util.UNBOUND_DATE_RANGE),
        util.add_unbound_time_to_logs_link('abc'))

  def test_download_command_from_logs_link_fail_to_parse(self):
    with self.assertRaises(ValueError):
      _ = util.download_command_from_logs_link('abc')

  def test_download_command_from_logs_link_success(self):
    self.assertEqual(
        VALID_DOWNLOAD_COMMAND,
        util.download_command_from_logs_link(VALID_LOGS_LINK))

  def test_test_name_from_logs_link_fail_to_parse(self):
    with self.assertRaises(ValueError):
      _ = util.test_name_from_logs_link('abc')

  def test_test_name_from_logs_link_success(self):
    self.assertEqual(
        VALID_TEST_NAME,
        util.test_name_from_logs_link(VALID_LOGS_LINK))


if __name__ == '__main__':
  unittest.main()
