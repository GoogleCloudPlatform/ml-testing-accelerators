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

import util

from absl import logging
from google.cloud import error_reporting
from google.cloud import secretmanager
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, From, To, Subject, PlainTextContent, HtmlContent

# Make sure these match the names of your Cloud Secrets. Refer to the README
# in this directory for email alert setup steps.
_SENDGRID_API_SECRET_NAME = 'sendgrid-api-key'
_RECIPIENT_EMAIL_SECRET_NAME = 'alert-destination-email-address'
_SENDER_EMAIL_SECRET_NAME = 'alert-sender-email-address'

# An error not specific to any test, e.g. a failure to read from Pubsub,
# will not have any corresponding training logs.
_NO_LOGS = 'no logs'


class AlertHandler(object):
  def __init__(self, project_id, write_to_logging=True,
               write_to_error_reporting=True, write_to_email=True):
    """Handles logging and alerting for each run of the Metrics Handler.

    Args:
      project_id (string): Name of your Cloud project.
      write_to_logging (bool, optional): If False, skip all logging.
      write_to_error_reporting (bool, optional): If False, do not report any
        errors to Stackdriver Error Reporting.
      write_to_email (bool, optional): If False, do not send any alert emails.
        See the README in this directory for how to set up email alerts.
    """
    self.project_id = project_id
    self.write_to_logging = write_to_logging
    self.write_to_error_reporting = write_to_error_reporting
    self.write_to_email = write_to_email
    if write_to_error_reporting:
      self.error_reporter = error_reporting.Client()
    if self.write_to_email:
      try:
        secret_client = secretmanager.SecretManagerServiceClient()
        api_key = self._get_secret_value(
            _SENDGRID_API_SECRET_NAME, secret_client)
        self.sendgrid = SendGridAPIClient(api_key)
        self.recipient_email = self._get_secret_value(
            _RECIPIENT_EMAIL_SECRET_NAME, secret_client)
        self.sender_email = self._get_secret_value(
            _SENDER_EMAIL_SECRET_NAME, secret_client)
        self.messages_to_email = collections.defaultdict(list)
        self.write_to_email = True
      except Exception as e:
        self._log('Failed to initialize alert email client. See '
                  'metrics_handler/README for setup steps. Error '
                  'was: {}'.format(e), logging.ERROR)
        self.write_to_email = False

  def _get_secret_value(self, secret_name, secret_client):
    secret_resource = \
      f'projects/{self.project_id}/secrets/{secret_name}/versions/latest'
    lookup_response = secret_client.access_secret_version(
        secret_resource)
    return lookup_response.payload.data.decode('UTF-8')

  def _log(self, message, log_level):
    logging.log(log_level, message)

  def _report_error(self, message, logs_link):
    if logs_link != _NO_LOGS:
      message += ' ||| Logs for this run: {}'.format(logs_link)
    self.error_reporter.report(message)

  def _add_to_email(self, message, logs_link):
    self.messages_to_email[logs_link].append(message)

  def _log_all(self, message, log_level, logs_link=_NO_LOGS):
    if self.write_to_logging:
      self._log(message, log_level)
    if self.write_to_error_reporting and log_level <= logging.ERROR:
      self._report_error(message, logs_link)
    if self.write_to_email and log_level <= logging.ERROR:
      self._add_to_email(message, logs_link)

  def debug(self, message):
    """Log a message at DEBUG level.

    Args:
      message (string): Message to log.
    """
    self._log_all(message, logging.DEBUG)

  def info(self, message):
    """Log a message at INFO level.

    Args:
      message (string): Message to log.
    """
    self._log_all(message, logging.INFO)

  def warning(self, message):
    """Log a message at WARNING level.

    Args:
      message (string): Message to log.
    """
    self._log_all(message, logging.WARNING)

  def error(self, message, logs_link=_NO_LOGS):
    """Log a message at ERROR level.

    This will also trigger a report to Stackdriver Error Reporting and add to
    an alert email draft.

    Args:
      message (string): Message to log.
      logs_link (string, Optional): Link to the Stackdriver Logs of the test
        where this message originated. If provided, will be included in the
        alert email and the Stackdriver Error.
    """
    self._log_all(message, logging.ERROR, logs_link=logs_link)

  def fatal(self, message, logs_link=_NO_LOGS):
    """Log a message at FATAL level.

    This will also trigger a report to Stackdriver Error Reporting and add to
    an alert email draft.

    Args:
      message (string): Message to log.
      logs_link (string, Optional): Link to the Stackdriver Logs of the test
        where this message originated. If provided, will be included in the
        alert email and the Stackdriver Error.
    """
    self._log_all(message, logging.FATAL, logs_link=logs_link)

  def send_email(self):
    """Sends alert email and clears the current email draft."""
    if not self.write_to_email or not self.messages_to_email:
      return
    html_message_body = 'New errors in test suite for {}:'.format(
        self.project_id)
    html_message_body += '<ul>'
    for logs_link in self.messages_to_email.keys():
      html_message_body += '<li>{}:'.format(
          'General errors' if logs_link == _NO_LOGS else \
              util.test_name_from_logs_link(logs_link))
      html_message_body += '<ul>'
      for message in self.messages_to_email[logs_link]:
        html_message_body += '<li>{}</li>'.format(message)

      # If the error was specific to a certain test, include links to quickly
      # access the logs from that test.
      if logs_link != _NO_LOGS:
        html_message_body += '<li><a href="{}">Stackdriver logs for this ' \
                             'run of the test</a></li>'.format(logs_link)
        html_message_body += '<li>Command to download plaintext logs: ' \
                             '<code style="background-color:#e3e3e3;">' \
                             '{}</code></li>'.format(
                                 util.download_command_from_logs_link(
                                     logs_link))
      html_message_body += '</ul>'
      html_message_body += '</li>'
    html_message_body += '</ul>'
    message = Mail(
        from_email=From(self.sender_email,
                        'Cloud Accelerators Alert Manager'),
        to_emails=[To(self.recipient_email)],
        subject=Subject('Errors in ML Accelerators Tests at {}'.format(
            datetime.now().strftime("%Y/%m/%d %H:%M:%S"))),
        plain_text_content=PlainTextContent('empty'),
        html_content=HtmlContent(html_message_body))
    response = self.sendgrid.send(message)
    self._log('Email send attempt response: {}\n{}'.format(
        response.status_code, response.headers), logging.INFO)
    self.messages_to_email.clear()
