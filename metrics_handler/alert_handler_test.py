# Lint as: python3
"""Tests for job_status_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
from absl.testing import parameterized
import alert_handler


VALID_LOGS_LINK = 'https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-fs-transformer-functional-v3-8-1585756800&extra_junk&dateRangeUnbound=backwardInTime'
HTML_GENERAL_ERRORS_ONLY = """New errors in test suite for my-project-id:<ul><li>General errors:<ul><li>msg1</li><li>msg2</li></ul></li></ul>"""
HTML_GENERAL_AND_SPECIFIC_ERRORS = """New errors in test suite for my-project-id:<ul><li>General errors:<ul><li>msg1</li><li>msg2</li></ul></li><li>pt-1.5-fs-transformer-functional-v3-8-1585756800:<ul><li>msg3</li><li>msg4</li><li><a href="https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-fs-transformer-functional-v3-8-1585756800&extra_junk&dateRangeUnbound=backwardInTime">Stackdriver logs for this run of the test</a></li><li><a href="https://console.cloud.google.com/kubernetes/job/us-central1-b/xl-ml-test/automated/pt-1.5-fs-transformer-functional-v3-8-1585756800?project=xl-ml-test&pt-1.5-fs-transformer-functional-v3-8-1585756800_events_tablesize=50&tab=events&duration=P30D&pod_summary_list_tablesize=20&service_list_datatablesize=20">Kubernetes workload for this run of the test</a></li><li>Command to download plaintext logs: <code style="background-color:#e3e3e3;">gcloud logging read 'resource.type=k8s_container resource.labels.project_id=xl-ml-test resource.labels.location=us-central1-b resource.labels.cluster_name=xl-ml-test resource.labels.namespace_name=automated resource.labels.pod_name:pt-1.5-fs-transformer-functional-v3-8-1585756800' --limit 10000000000000 --order asc --format 'value(textPayload)' --project=xl-ml-test > pt-1.5-fs-transformer-functional-v3-8-1585756800_logs.txt && sed -i '/^$/d' pt-1.5-fs-transformer-functional-v3-8-1585756800_logs.txt</code></li></ul></li></ul>"""
HTML_SPECIFIC_ERRORS_ONLY = """New errors in test suite for my-project-id:<ul><li>pt-1.5-fs-transformer-functional-v3-8-1585756800:<ul><li>msg3</li><li>msg4</li><li><a href="https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-fs-transformer-functional-v3-8-1585756800&extra_junk&dateRangeUnbound=backwardInTime">Stackdriver logs for this run of the test</a></li><li><a href="https://console.cloud.google.com/kubernetes/job/us-central1-b/xl-ml-test/automated/pt-1.5-fs-transformer-functional-v3-8-1585756800?project=xl-ml-test&pt-1.5-fs-transformer-functional-v3-8-1585756800_events_tablesize=50&tab=events&duration=P30D&pod_summary_list_tablesize=20&service_list_datatablesize=20">Kubernetes workload for this run of the test</a></li><li>Command to download plaintext logs: <code style="background-color:#e3e3e3;">gcloud logging read 'resource.type=k8s_container resource.labels.project_id=xl-ml-test resource.labels.location=us-central1-b resource.labels.cluster_name=xl-ml-test resource.labels.namespace_name=automated resource.labels.pod_name:pt-1.5-fs-transformer-functional-v3-8-1585756800' --limit 10000000000000 --order asc --format 'value(textPayload)' --project=xl-ml-test > pt-1.5-fs-transformer-functional-v3-8-1585756800_logs.txt && sed -i '/^$/d' pt-1.5-fs-transformer-functional-v3-8-1585756800_logs.txt</code></li></ul></li></ul>"""


class AlertHandlerTest(parameterized.TestCase):
  def setUp(self):
    self.handler = alert_handler.AlertHandler(
      project_id='my-project-id',
      write_to_logging=True,
      write_to_error_reporting=False,
      write_to_email=False  # Skip the init of the real email client.
    )
    self.handler.write_to_email = True  # Enable writing to draft email.
    # The step below normally would have happened while initializing the
    # real email client.
    self.handler.messages_to_email = collections.defaultdict(list)

  @parameterized.named_parameters(
      ('debug', 'DEBUG', 'debug'),
      ('info', 'INFO', 'info'),
      ('warning', 'WARNING', 'warning'),
  )
  def test_log_no_email(self, log_level, log_method_name):
    with self.assertLogs(level=log_level) as cm:
      getattr(self.handler, log_method_name)('log message')
      self.assertEqual(cm.output, [f'{log_level}:absl:log message'])
    self.assertEmpty(self.handler.messages_to_email)

  @parameterized.named_parameters(
      ('error', 'ERROR', 'error'),
      ('fatal', 'CRITICAL', 'fatal'),
  )
  def test_log_with_email(self, log_level, log_method_name):
    with self.assertLogs() as cm:
      getattr(self.handler, log_method_name)('msg1')
      getattr(self.handler, log_method_name)('msg2')
      getattr(self.handler, log_method_name)('msg3', logs_link='link1')
      getattr(self.handler, log_method_name)('msg4', logs_link='link1')
      getattr(self.handler, log_method_name)('msg5', logs_link='link2')
      self.assertEqual(cm.output, [
          f'{log_level}:absl:msg1',
          f'{log_level}:absl:msg2',
          f'{log_level}:absl:msg3',
          f'{log_level}:absl:msg4',
          f'{log_level}:absl:msg5',
      ])
    expected_messages = {
        alert_handler._NO_LOGS: ['msg1', 'msg2'],
        'link1': ['msg3', 'msg4'],
        'link2': ['msg5'],
    }
    self.assertDictEqual(
        self.handler.messages_to_email,
        expected_messages)

  def test_generate_email_body_no_messages(self):
    self.assertEqual(self.handler.generate_email_body(), '')

  def test_generate_email_body_general_errors_only(self):
    self.handler.messages_to_email = {
        alert_handler._NO_LOGS: ['msg1', 'msg2'],
    }
    self.assertEqual(
        self.handler.generate_email_body(),
        HTML_GENERAL_ERRORS_ONLY)

  def test_generate_email_body_specific_errors_only(self):
    self.handler.messages_to_email = {
        VALID_LOGS_LINK: ['msg3', 'msg4'],
    }
    self.assertEqual(
        self.handler.generate_email_body(),
        HTML_SPECIFIC_ERRORS_ONLY)

  def test_generate_email_body_general_and_specific_errors(self):
    self.handler.messages_to_email = {
        alert_handler._NO_LOGS: ['msg1', 'msg2'],
        VALID_LOGS_LINK: ['msg3', 'msg4'],
    }
    self.assertEqual(
        self.handler.generate_email_body(),
        HTML_GENERAL_AND_SPECIFIC_ERRORS)


if __name__ == '__main__':
  absltest.main()
