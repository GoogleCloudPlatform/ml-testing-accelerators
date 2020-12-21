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

import base64
import itertools
import os
import typing
import uuid

from absl import logging
import google.auth
import numpy as np
import sendgrid
from sendgrid.helpers import mail

import alerts
import bigquery_client
import collectors
import utils
import metrics_pb2

try:
  DATASET = os.environ['BQ_DATASET']
except KeyError:
  raise KeyError('Must set $BQ_DATASET env var.')

SEND_EMAIL_ALERTS = os.getenv('SEND_EMAIL_ALERTS', None)
PROJECT = os.getenv('GCP_PROJECT', None)

SOURCE_TO_COLLECTOR = {
  'literals': collectors.literal.LiteralCollector,
  'perfzero': collectors.perfzero.PerfZeroCollector,
  'tensorboard': collectors.tensorboard.TensorBoardCollector,
}

# Make sure these match the names of your Cloud Secrets. Refer to the README
# in this directory for email alert setup steps.
_SENDGRID_API_SECRET_NAME = 'sendgrid-api-key'
_RECIPIENT_EMAIL_SECRET_NAME = 'alert-destination-email-address'
_SENDER_EMAIL_SECRET_NAME = 'alert-sender-email-address'

def _send_email(project_id: str, subject: mail.Subject, body: mail.HtmlContent):
  secret_client = secretmanager.SecretManagerServiceClient()

  def _get_secret_value(secret_name):
    secret_resource = \
      f'projects/{project_id}/secrets/{secret_name}/versions/latest'
    lookup_response = secret_client.access_secret_version(
        secret_resource)
    return lookup_response.payload.data.decode('UTF-8')

  try:
    api_key = _get_secret_value(_SENDGRID_API_SECRET_NAME)
    recipient_email = _get_secret_value(_RECIPIENT_EMAIL_SECRET_NAME)
    sender_email = _get_secret_value(_SENDER_EMAIL_SECRET_NAME)
    sendgrid = sendgrid.SendGridAPIClient(api_key)
  except Exception as e:
    logging.error(
        'Failed to initialize alert email client. See'
        'metrics/handler/README.md for setup steps.', exc_info=e)
    return True

  message = Mail(
      from_email=mail.From(sender_email,
                            'Cloud Accelerators Alert Manager'),
      to_emails=[mail.To(recipient_email)],
      subject=subject,
      plain_text_content=PlainTextContent('empty'),
      html_content=body)

  try:
    response = sendgrid.send(message)
    logging.info('Email send attempt response: %s\n%s',
        str(response.status_code), str(response.headers))
  except Exception as e:
    logging.error('Failed to send e-mail.', exc_info=e)


def process_proto_message(
      event: metrics_pb2.TestCompletedEvent,
      metric_store: bigquery_client.BigQueryMetricStore,
      message_id: typing.Optional[str] = None,
  ) -> typing.Tuple[bigquery_client.JobHistoryRow, typing.Iterable[bigquery_client.MetricHistoryRow]]:
  """Collects test status and metrics and writes to BigQuery.

  Args:
    event: Parsed TestCompletedEvent proto.
    project: GCP project id.
    dataset: BigQuery dataset to read and store metric data.
    message_id: Unique message ID to match jobs to metrics in BQ.

  Returns:
    Job status and metrics to be inserted into BigQuery. Metrics may be empty
    if the job failed or no metrics were collected.
  """

  unique_key = message_id or str(uuid.uuid4())
  job_row = bigquery_client.JobHistoryRow.from_test_event(unique_key, event)

  # Alert for failing jobs unless the user has explicitly added a config
  # that disables alerts for this test.
  completed = event.status == metrics_pb2.TestCompletedEvent.COMPLETED
  if not completed:
    if not event.metric_collection_config.silence_alerts:
      logging.error(
          'job_status was `%s` for test `%s`',
          metrics_pb2.TestCompletedEvent.TestStatus.Name(event.status),
          event.benchmark_id)
    if not event.metric_collection_config.record_failing_test_metrics:
      return job_row, []

  collectors = []
  for source in event.metric_collection_config.sources:
    source_type = source.WhichOneof('source_type')
    collector = SOURCE_TO_COLLECTOR[source_type](event, source, metric_store)
    collectors.append(collector)

  metrics = set(itertools.chain.from_iterable(c.metric_points() for c in collectors))

  metric_rows = [
      bigquery_client.MetricHistoryRow.from_metric_point(unique_key, point, event)
      for point in metrics
  ]

  for metric in metrics:
    if not metric.within_bounds():
      logging.error(
          'Metric %s was out of bounds for test %s. Bounds were (%.2f, %.2f) '
          'and value was %.2f', metric.metric_key, event.benchmark_id,
          metric.bounds.lower, metric.bounds.upper, metric.metric_value)

  return job_row, metric_rows

def receive_test_event(data: dict, context: dict) -> bool:
  """Entrypoint for Cloud Function.

  Args:
    data: dict containing base64-encoded proto message.
    context: dict containing event metadata.

  Returns:
    True if message should be ack-ed, else False.
  """
  logging.set_verbosity(logging.INFO)

  dataset = DATASET
  project = PROJECT or google.auth.default()[1]

  try:
    message_bytes = base64.b64decode(data['data'])
    event = metrics_pb2.TestCompletedEvent()
    event.ParseFromString(message_bytes)
  except Exception as e:
    logging.fatal('Failed to parse PubSub message. Will ack message to prevent '
                  'more crashes.', exc_info=e)
    return True

  alert_handler = (
      alerts.AlertHandler(project, event.benchmark_id, event.debug_info, level='ERROR'))
  logging.get_absl_logger().addHandler(alert_handler)

  metric_store = bigquery_client.BigQueryMetricStore(
    project=project,
    dataset=dataset,
  )
  try:
    logging.info('Processing test event: %s', str(event))
    job_row, metric_rows = process_proto_message(
        event, metric_store, context.event_id)
    metric_store.insert_status_and_metrics(job_row, metric_rows)
  except Exception as e:
    logging.fatal(
        'Encountered exception while attempting to process message.',
        exc_info=e)

  if alert_handler.has_errors:
    logging.info('Alerts: %s', str(alert_handler._records))
    if SEND_EMAIL_ALERTS:
      _send_email(project, *alert_handler.generate_email_content)
    else:
      logging.info('E-mail alerts disabled.')
  else:
    logging.info('No alerts found.')
      
  return True
