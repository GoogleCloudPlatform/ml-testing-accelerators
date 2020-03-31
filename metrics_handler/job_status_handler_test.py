# Lint as: python3
"""Tests for job_status_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time
import types

from absl.testing import absltest
from absl.testing import parameterized
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


class JobStatusHandlerTest(parameterized.TestCase):
  def setUp(self):
    self.logger = alert_handler.AlertHandler(
      project_id=None,
      write_to_logging=True,
      write_to_error_reporting=False,
      write_to_email=False)
    self.handler = job_status_handler.JobStatusHandler(
        "unused", "unused", "unused", self.logger)

  @parameterized.named_parameters(
    ('success_with_completion_time', {
        'succeeded': 1,
        'num_failures': 1,
        'completion_time': datetime.datetime(2020, 3, 30, 17, 46, 15),
        'expected_status': job_status_handler.SUCCESS}),
    ('success_with_condition_time', {
        'succeeded': 1,
        'num_failures': 1,
        'condition_transition_time': datetime.datetime(2020, 3, 30, 17, 46, 15),
        'condition_reason': 'DeadlineExceeded',
        'expected_status': job_status_handler.SUCCESS}),
    ('timeout', {
        'num_failures': 1,
        'condition_transition_time': datetime.datetime(2020, 3, 30, 17, 46, 15),
        'condition_reason': 'DeadlineExceeded',
        'expected_status': job_status_handler.TIMEOUT}),
    ('failure', {
        'num_failures': 1,
        'condition_transition_time': datetime.datetime(2020, 3, 30, 17, 46, 15),
        'condition_reason': 'BackoffLimitExceeded',
        'expected_status': job_status_handler.FAILURE}),
  )
  def test_interpret_status(self, args_dict):
    status = FakeKubernetesStatus(
      succeeded=args_dict.get('succeeded'),
      num_failures=args_dict.get('num_failures'),
      completion_time=args_dict.get('completion_time'),
      condition_transition_time=args_dict.get('condition_transition_time'),
      condition_reason=args_dict.get('condition_reason'),
    )
    status_code, stop_time, num_failures = self.handler.interpret_status(
        status, "my_job_name")
    self.assertEqual(status_code, args_dict['expected_status'])
    expected_datetime = args_dict.get('completion_time') or args_dict[
        'condition_transition_time']
    self.assertEqual(stop_time, expected_datetime.timestamp())
    self.assertEqual(num_failures, args_dict.get('num_failures', 0))

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


if __name__ == '__main__':
  absltest.main()
