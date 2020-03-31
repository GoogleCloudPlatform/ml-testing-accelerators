# Lint as: python3
"""Tests for job_status_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time
import types

from absl.testing import absltest
import alert_handler
import job_status_handler


class FakeKubernetesStatus(object):
  def __init__(self, active=None, succeeded=None, num_failures=None,
               start_time=None, completion_time=None,
               condition_transition_time=None, condition_reason=None):
    self.active = active
    self.succeeded = succeeded
    self.failed = num_failures
    self.start_time = start_time
    self.completion_time = completion_time
    if condition_transition_time:
      condition = types.SimpleNamespace()  # Simple object with attrs.
      condition.last_transition_time = condition_transition_time
      condition.reason = condition_reason
      self.conditions = [condition]
    else:
      self.conditions = None


class JobStatusHandlerTest(absltest.TestCase):
  def setUp(self):
    self.logger = alert_handler.AlertHandler(
      project_id=None,
      write_to_logging=True,
      write_to_error_reporting=False,
      write_to_email=False)
    self.handler = job_status_handler.JobStatusHandler(
        "unused", "unused", "unused", self.logger)

  def test_interpret_status_success_with_completion_time(self):
    expected_num_failures = 1
    expected_stop_time = datetime.datetime(
        2020, 3, 30, 17, 46, 15)
    status = FakeKubernetesStatus(
        succeeded=1,
        num_failures=1,
        completion_time=expected_stop_time)
    status_code, stop_time, num_failures = self.handler.interpret_status(
        status, "my_job_name")
    self.assertEqual(status_code, job_status_handler.SUCCESS)
    self.assertEqual(stop_time, expected_stop_time.timestamp())
    self.assertEqual(num_failures, expected_num_failures)

  def test_interpret_status_success_without_completion_time(self):
    expected_num_failures = 1
    expected_stop_time = datetime.datetime(
        2020, 3, 30, 17, 46, 15)
    status = FakeKubernetesStatus(
        succeeded=1,
        num_failures=1,
        condition_transition_time=expected_stop_time,
        condition_reason='DeadlineExceeded')
    status_code, stop_time, num_failures = self.handler.interpret_status(
        status, "my_job_name")
    self.assertEqual(status_code, job_status_handler.SUCCESS)
    self.assertEqual(stop_time, expected_stop_time.timestamp())
    self.assertEqual(num_failures, expected_num_failures)

  def test_interpret_status_success_without_completion_or_transition(self):
    expected_num_failures = 1
    now = time.time()
    status = FakeKubernetesStatus(
        succeeded=1,
        num_failures=1)
    status_code, stop_time, num_failures = self.handler.interpret_status(
        status, "my_job_name")
    self.assertEqual(status_code, job_status_handler.SUCCESS)
    self.assertGreaterEqual(stop_time, now)
    self.assertEqual(num_failures, expected_num_failures)

  def test_interpret_status_timeout(self):
    expected_num_failures = 1
    expected_stop_time = datetime.datetime(
        2020, 3, 30, 17, 46, 15)
    status = FakeKubernetesStatus(
        num_failures=1,
        condition_transition_time=expected_stop_time,
        condition_reason='DeadlineExceeded')
    status_code, stop_time, num_failures = self.handler.interpret_status(
        status, "my_job_name")
    self.assertEqual(status_code, job_status_handler.TIMEOUT)
    self.assertEqual(stop_time, expected_stop_time.timestamp())
    self.assertEqual(num_failures, expected_num_failures)

  def test_interpret_status_failure(self):
    expected_num_failures = 1
    expected_stop_time = datetime.datetime(
        2020, 3, 30, 17, 46, 15)
    status = FakeKubernetesStatus(
        num_failures=1,
        condition_transition_time=expected_stop_time,
        condition_reason='BackoffLimitExceeded')
    status_code, stop_time, num_failures = self.handler.interpret_status(
        status, "my_job_name")
    self.assertEqual(status_code, job_status_handler.FAILURE)
    self.assertEqual(stop_time, expected_stop_time.timestamp())
    self.assertEqual(num_failures, expected_num_failures)


if __name__ == '__main__':
  absltest.main()
