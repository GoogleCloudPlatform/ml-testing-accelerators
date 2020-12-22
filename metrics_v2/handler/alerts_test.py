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
from datetime import datetime

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import pytz

import alerts
import metrics_pb2


LOGS_LINK = """https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-fs-transformer-functional-v3-8-1585756800&extra_junk&dateRangeUnbound=backwardInTime"""
WORKLOAD_LINK = """https://console.cloud.google.com/kubernetes/job/us-central1-b/xl-ml-test/automated/pt-1.5-fs-transformer-functional-v3-8-1585756800?project=xl-ml-test&pt-1.5-fs-transformer-functional-v3-8-1585756800_events_tablesize=50&tab=events&duration=P30D&pod_summary_list_tablesize=20&service_list_datatablesize=20"""

class AlertHandlerTest(parameterized.TestCase):
  def setUp(self):
    self._logger = logging.get_absl_logger()
    self._handler = alerts.AlertHandler(
      project_id='my-project-id',
      benchmark_id='benchmark-id',
      debug_info=None,
    )
    self._logger.addHandler(self._handler)

  def tearDown(self):
    self._logger.removeHandler(self._handler)

  @parameterized.named_parameters(
      ('debug', 'DEBUG', logging.debug),
      ('info', 'INFO', logging.info),
      ('warning', 'WARNING', logging.warning),
  )
  def test_log_no_email(self, log_level, log_method):
    self._handler.setLevel('ERROR')
    with self.assertLogs(level=log_level) as cm:
      log_method('log message')
      self.assertEqual(cm.output, [f'{log_level}:absl:log message'])
    self.assertEmpty(self._handler._records)

  @parameterized.named_parameters(
      ('error', 'ERROR', logging.error),
      ('fatal', 'CRITICAL', logging.fatal),
  )
  def test_log_with_email(self, log_level, log_method):
    self._handler.setLevel('ERROR')
    with self.assertLogs() as cm:
      log_method('msg1')
      log_method('msg2')
      log_method('msg3')
      self.assertEqual(cm.output, [
          f'{log_level}:absl:msg1',
          f'{log_level}:absl:msg2',
          f'{log_level}:absl:msg3',
      ])
    
    subject, body = self._handler.generate_email_content()
    self.assertIn('msg1', body.content)
    self.assertIn('msg2', body.content)
    self.assertIn('msg3', body.content)

  def test_has_errors_when_empty(self):
    self.assertFalse(self._handler.has_errors)
    self._handler.generate_email_content()

  def test_generate_email_body_with_debug_info(self):
    self._handler._debug_info = metrics_pb2.DebugInfo(
      logs_link=LOGS_LINK,
      details_link=WORKLOAD_LINK,
    )
    logging.error('error_message')
    _, body = self._handler.generate_email_content()
    self.assertIn('error_message', body.content)
    self.assertIn(LOGS_LINK, body.content)
    self.assertIn(WORKLOAD_LINK, body.content)

  def test_generate_email_subject(self):
    subject = self._handler.generate_email_content()[0].subject
    date_str = subject[subject.find('202'):] # Grab substring from start of year onward.
    tz = pytz.timezone('US/Pacific')
    sj_date = tz.localize(datetime.strptime(date_str, '%Y/%m/%d %H:%M:%S'))
    after_call = datetime.now(tz)
    # This checks that the datetime string used in the email is 1. using
    # current time and 2. is using US/Pacific tz and not e.g. UTC.
    self.assertTrue((after_call - sj_date).total_seconds() < 2.0)

    self.assertIn('benchmark-id', subject)

  def test_no_html_injection(self):
    injection_str = '<marquee>HTML injection is fun!</marquee>'
    logging.error(injection_str)

    _, body = self._handler.generate_email_content()
    self.assertNotIn(injection_str, body.content)

if __name__ == '__main__':
  absltest.main()
