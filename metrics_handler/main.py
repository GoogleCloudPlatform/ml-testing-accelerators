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
import collections
import datetime
import io
import itertools
import json
import math
import time
import traceback
import uuid

import alert_handler
import job_status_handler
import metrics
import util

import google.api_core.exceptions
from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.cloud import storage as gcs
import numpy as np
from tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf


BQ_DATASET_NAME = 'metrics_handler_dataset'
BQ_JOB_TABLE_NAME = 'job_history'
BQ_METRIC_TABLE_NAME = 'metric_history'
METRICS_WRITTEN_TOPIC = 'metrics-written'
MIN_MSG_AGE_SEC = 60


class CloudMetricsHandler(object):
  def __init__(self, test_name, events_dir, debug_info,
               metric_collection_config, regression_test_config,
               test_type, accelerator, framework_version, logger):
    """Handles metrics storage, collection, aggregation, alerts, etc.

    Used in conjunction with the Cloud Accelerator Testing framework

    Args:
      test_name (string): This should be a string that uniquely identifies
        a test among all your other Cloud Accelerator tests. Used for some
        naming, e.g. for error strings if this test's metrics regress.
      events_dir (string): Path to a GCS or local directory where Tensorboard
        summaries are stored.
      debug_info (alert_handler.DebugInfo): Should include job_name, link to
        stackdriver logs, command to download plaintext logs, and link to the
        Kubernetes workload for this run of the test.
      metric_collection_config (dict): Options for collecting metrics. See
        README for documentation.
      regression_test_config (dict): Options for alerting in the event of
        metrics regressions. See README for documentation.
      test_type (string): E.g. 'conv' or 'func'. Used to organize metrics in
        Bigquery.
      accelerator (string): E.g. 'tpu-v2-8'. The type of accelerator used to
        run this test. Used to organize metrics in Bigquery.
      framework_version (string): E.g. 'pt-nightly' or 'tf-nightly'. The
        combined ML framework + version identifier used to run this test. Used
        to organize metrics in Bigquery.
      logger (`AlertHandler` instance): Used to write logs and alert emails.
    """
    self.test_name = test_name
    self.events_dir = events_dir
    self.debug_info = debug_info
    self.metric_collection_config = metric_collection_config or {}
    self.regression_test_config = regression_test_config or {}
    self.test_type = test_type
    self.accelerator = accelerator
    self.framework_version = framework_version
    self.logger = logger

    if self.metric_collection_config.get('write_to_bigquery'):
      # Initalize clients to interact with various Cloud APIs.
      self.project = google.auth.default()[1]
      self.bigquery_client = bigquery.Client()
      self.gcs_client = gcs.Client()

      self.job_history_table_id = self._get_table_id(
        BQ_DATASET_NAME, BQ_JOB_TABLE_NAME)
      self.metric_history_table_id = self._get_table_id(
          BQ_DATASET_NAME, BQ_METRIC_TABLE_NAME)
      self._make_bigquery_tables()

  @staticmethod
  def _wall_time_to_sql_timestamp(wall_time):
    return datetime.datetime.fromtimestamp(wall_time).strftime(
        '%Y-%m-%d %H:%M:%S')

  def _get_table_id(self, dataset_name, table_name):
    return '{}.{}.{}'.format(self.project, dataset_name, table_name)

  def get_metrics_from_events_dir(self):
    """Retrieves and aggregates metrics from Tensorboard Summary file.

    Returns:
      final_metrics (dict): Key is metric name and value is a MetricPoint
        containing the aggregated value for that metric.
    """
    tags_to_ignore = set(
        self.metric_collection_config.get('tags_to_ignore', []))
    raw_metrics = metrics.read_metrics_from_events_dir(
        self.events_dir, tags_to_ignore)

    if not raw_metrics:
      self.logger.warning("No metrics found in {}".format(self.events_dir))
      return {}

    default_aggregation_strategies = self.metric_collection_config.get(
        'default_aggregation_strategies')
    metric_to_aggregation_strategies = self.metric_collection_config.get(
        'metric_to_aggregation_strategies')
    try:
      final_metrics = metrics.aggregate_metrics(
          raw_metrics,
          default_aggregation_strategies,
          metric_to_aggregation_strategies
      )
    except ValueError as e:
      raise ValueError("Error during metric aggregation: {}".format(e))
    return final_metrics

  def add_computed_metrics(self, metrics_dict, job_status_dict,
                           find_memory_metrics=True):
    """Computes additional metrics and adds them to `metrics_dict`.

    Args:
      metrics_dict (dict): Keys are strings and values are MetricPoints.
      job_status_dict (dict): Should contain `job_status`, `start_time`,
        and `stop_time` as keys.
      find_memory_metrics (bool, optional): If True, query Cloud Monitoring
        to find memory usage metrics and add them to `metrics_dict`.
    """
    start_time = job_status_dict['start_time']
    stop_time = job_status_dict['stop_time']
    metrics_dict['total_wall_time'] = metrics.MetricPoint(
        stop_time - start_time, stop_time)

    # Compute time_to_accuracy if requested in the config.
    tta_config = self.metric_collection_config.get('time_to_accuracy')
    if tta_config:
      if 'accuracy_tag' not in tta_config or \
          'accuracy_threshold' not in tta_config:
        raise ValueError('Invalid `time_to_accuracy` portion of config. '
                         'See README for how to set up the config.')
      tag = tta_config['accuracy_tag']
      threshold = tta_config['accuracy_threshold']
      try:
        metrics_dict['time_to_accuracy'] = metrics.time_to_accuracy(
            metrics_dict, tag, threshold)
      except ValueError as e:
        raise ValueError('Error computing time to accuracy: {}'.format(e))

    if find_memory_metrics:
      metrics.compute_memory_metrics(metrics_dict, self.project,
                                     self.debug_info.job_name)


  def _make_bigquery_tables(self):
    if not self.metric_collection_config.get('write_to_bigquery'):
      return
    dataset = bigquery.Dataset(self.bigquery_client.dataset(BQ_DATASET_NAME))
    _ = self.bigquery_client.create_dataset(dataset, exists_ok=True)

    job_history_schema = [
        bigquery.SchemaField("uuid", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("test_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("test_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("accelerator", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("framework_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("job_status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("num_failures", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("job_duration_sec", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("stackdriver_logs_link", "STRING",
                             mode="REQUIRED"),
        bigquery.SchemaField("msg_publish_time", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("logs_download_command", "STRING",
                             mode="NULLABLE"),
        bigquery.SchemaField("kubernetes_workload_link", "STRING",
                             mode="NULLABLE"),
    ]
    job_history_table = bigquery.Table(
        self.job_history_table_id, schema=job_history_schema)
    _ = self.bigquery_client.create_table(job_history_table, exists_ok=True)

    metric_history_schema = [
        bigquery.SchemaField("uuid", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("test_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_value", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("metric_lower_bound", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("metric_upper_bound", "FLOAT64", mode="NULLABLE"),
    ]
    metric_history_table = bigquery.Table(
        self.metric_history_table_id, schema=metric_history_schema)
    _ = self.bigquery_client.create_table(metric_history_table, exists_ok=True)


  def add_status_and_metrics_to_bigquery(
      self, job_status, aggregated_metrics_dict,
      metric_name_to_visual_bounds=None):
    """Adds job_status and metrics to their respective BigQuery tables.

    Args:
      job_status (dict): Contains information about the Kubernetes Job.
      aggregated_metrics_dict (dict): Key is metric name and value is a
        MetricPoint containing the aggregated value for that metric.
      metric_name_to_visual_bounds (dict, optional): Key is metric name and
        value is a tuple consisting of (lower_bound, upper bound) for that
        metric. These bounds are useful for showing helpful lines on charts
        of metrics history. If not provided, the rows saved to BigQuery will
        have `null` for upper_bound and lower_bound.
    """
    if not self.metric_collection_config.get('write_to_bigquery', True):
      self.logger.info('Skipping writing metrics and job_status to BigQuery.')
      return

    if not metric_name_to_visual_bounds:
      metric_name_to_visual_bounds = {}

    # Compute the join key to link together job_history and metric_history.
    unique_key = str(uuid.uuid4())

    # Every job should have 1 job status row and it should exist even if
    # no other metrics exist.
    job_history_row = [
        unique_key,
        self.test_name,
        self.test_type,
        self.accelerator,
        self.framework_version,
        job_status['final_status'],
        job_status['num_failures'],
        int(job_status['stop_time'] - job_status['start_time']),
        self._wall_time_to_sql_timestamp(job_status['stop_time']),
        self.debug_info.stackdriver_logs_link,
        job_status['publish_time'],
        self.debug_info.download_command,
        self.debug_info.workload_link,
    ]

    # Create rows to represent the computed metrics for this job.
    metric_history_rows = []
    for metric_name, metric_point in aggregated_metrics_dict.items():
      lower_bound, upper_bound = metric_name_to_visual_bounds.get(
          metric_name, (None, None))
      metric_history_rows.append([
        unique_key,
        self.test_name,
        self._wall_time_to_sql_timestamp(metric_point.wall_time),
        metric_name,
        float(metric_point.metric_value),
        lower_bound,
        upper_bound,
      ])

    # Insert rows in Bigquery.
    for table_id, rows in [
        (self.job_history_table_id, [job_history_row]),
        (self.metric_history_table_id, metric_history_rows),
    ]:
      if not rows:
        continue
      self.logger.info(
          'Inserting {} rows into BigQuery table `{}`'.format(
              len(rows), table_id))
      table = self.bigquery_client.get_table(table_id)
      clean_rows = [util.replace_invalid_values(row) for row in rows]
      errors = self.bigquery_client.insert_rows(table, clean_rows)
      if errors == []:
        self.logger.info('Successfully added rows to Bigquery.')
      else:
        # TODO: Maybe add retry logic. insert_rows seems to be atomic for all
        #       elements in the list, so it should be safe to retry.
        self.logger.error(
            'Failed to add rows to Bigquery. Errors: {}'.format(errors),
            debug_info = self.debug_info)


  def get_metrics_history_from_bigquery(self):
    """Returns the historic values of each metric for a given model."""
    query_result = self.bigquery_client.query("""
        SELECT *
        FROM `{}`
        WHERE test_name like \"{}\" AND uuid in (
            SELECT uuid
            FROM `{}`
            WHERE test_name like \"{}\" AND job_status = \"success"\
        )""".format(
            self.metric_history_table_id,
            self.test_name,
            self.job_history_table_id,
            self.test_name)).result()
    metrics_history = collections.defaultdict(list)
    for row in query_result:
      metrics_history[row['metric_name']].append(metrics.MetricPoint(
          metric_value=row['metric_value'],
          wall_time=row['timestamp'].timestamp()))
    return metrics_history

  def get_existing_row(self):
    """Returns any existing row in job_history that is for the current test.

    Returns:
      uuid (string): The `uuid` column for the row. If no row exists,
        this will be None.
      publish_time (int): The `publish_time` column for the row. If no row
        exists, this will be None.
    """
    uuid = None
    publish_time = None
    if not self.metric_collection_config.get('write_to_bigquery', True):
      self.logger.info('Skipping check for existing Bigquery rows.')
      return uuid, publish_time
    query_result = self.bigquery_client.query(
        'SELECT * FROM `{}` WHERE stackdriver_logs_link=\"{}\"'.format(
            self.job_history_table_id,
            self.debug_info.stackdriver_logs_link)).result()
    if query_result.total_rows > 1:
      self.logger.error('Found more than 1 row in job_history for test: '
                        '{}'.format(self.test_name),
                        debug_info=self.debug_info)
    for row in query_result:
      uuid = row['uuid']
      publish_time = row['msg_publish_time']
    return uuid, publish_time

  def compute_bounds_and_report_errors(self, metrics_history, new_metrics,
                                       job_status):
    """Compute the bounds for metrics and report abnormal values.

    Any metric that is currently outside the expected bounds is reported to
    Stackdriver Error Reporting unless `write_to_error_reporting` is set to
    False in the regression test config. Even if this reporting is turned off,
    this method computes the upper and lower bounds for each metric to provide
    to BigQuery as a visual aid when rendering metrics history into charts.

    Args:
      metrics_history(dict): Historic values of each metric.
      new_metrics(dict): Key is metric name and value is MetricPoint containing
        the latest aggregated value for that metric.
      job_status(string): Final state of the job, should be one of the status
        constants found in job_status_handler.py.

    Returns:
      metric_name_to_visual_bounds (dict): Key is metric name and value is a
        tuple of floats of the form (lower_bound, upper_bound).
    """
    if not self.regression_test_config:
      return {}
    success_conditions = self.regression_test_config.get(
        'metric_success_conditions')
    if not success_conditions:
      return {}

    metrics_history = metrics_history.copy()

    # Add the metrics from the latest run. These aren't in Bigquery yet.
    for metric_name, metric_value in new_metrics.items():
      metrics_history[metric_name].append(metric_value)

    metric_name_to_visual_bounds = {}
    metric_subset_to_report = set(
        self.regression_test_config.get('metric_subset_to_alert', []))
    for metric_name, value_history in metrics_history.items():
      if metric_subset_to_report and metric_name not in metric_subset_to_report:
        self.logger.info(
            'Skipping alerts and bounds for metric `{}` since '
            'it does not appear in `metric_subset_to_report` in your '
            'regression test config.'.format(metric_name))
        continue
      success_condition = success_conditions.get(metric_name) or \
        success_conditions.get('default')
      if not success_condition:
        self.logger.warning(
            'metric: `{}` has an empty success condition in the '
            '`metric_success_conditions` dict in the regression_test_config '
            'but there is no default condition provided. No bounds or '
            'alerts will be computed. See README for config details.'.format(
                metric_name))
        continue
      elif len(value_history) <= success_condition.get(
        'wait_for_n_points_of_history', -1):
        self.logger.info(
            'Metric: {} had only {} points of history. Skipping bounds '
            'enforcement. Success condition: {}'.format(
                metric_name, len(value_history), success_condition))
        continue

      threshold_type, threshold_value = list(success_condition.get('success_threshold').items())[0]
      threshold = metrics.Threshold(threshold_type, threshold_value)
      comparison = success_condition.get('comparison')
      lower_bound, upper_bound = metrics.metric_bounds(
          value_history, threshold, comparison)
      metric_name_to_visual_bounds[metric_name] = (lower_bound, upper_bound)

      metric_value = value_history[-1].metric_value
      within_bounds = metrics.within_bounds(metric_value, lower_bound, upper_bound, inclusive=('equal' in comparison))

      # Generate an alert unless one of these is True:
      #   1. metrics are within bounds.
      #   2. alerting is disabled by config.
      #   3. the job failed and therefore metrics are unreliable.
      if within_bounds or not self.regression_test_config.get(
          'write_to_error_reporting', True) or \
              job_status != job_status_handler.SUCCESS:
        continue
      self.logger.error(
          'Metric `{}` was out of bounds for test `{}`. Bounds were '
          '({}, {}) and value was {:.2f}'.format(
              metric_name, self.test_name, lower_bound, upper_bound,
              metric_value),
          debug_info=self.debug_info)

    return metric_name_to_visual_bounds


def _process_pubsub_message(msg, status_handler, logger):
  publish_time = msg['publish_time']
  msg_age_sec = time.time() - publish_time
  if msg_age_sec < MIN_MSG_AGE_SEC:
    logger.warning('Message was {} seconds old, which is less than the '
                   'minimum of {}. Skipping for now but will retry on '
                   'the next run.'.format(msg_age_sec, MIN_MSG_AGE_SEC))
    return False  # Do not ack the message.
  events_dir = msg.get('model_dir')
  test_name = msg.get('test_name')
  logs_link = util.add_unbound_time_to_logs_link(msg.get('logs_link', ''))
  metric_collection_config = msg.get('metric_collection_config')
  regression_test_config = msg.get('regression_test_config')
  job_name = msg.get('job_name')
  job_namespace = msg.get('job_namespace')
  test_type = msg.get('test_type')
  accelerator = msg.get('accelerator')
  framework_version = msg.get('framework_version')
  zone = msg.get('zone')
  cluster = msg.get('cluster_name')
  project = google.auth.default()[1]
  download_command = util.download_command(
      job_name, job_namespace, zone, cluster, project)
  workload_link = util.workload_link(
      job_name, job_namespace, zone, cluster, project)
  debug_info = alert_handler.DebugInfo(
      job_name, logs_link, download_command, workload_link)

  if not (events_dir and test_name and logs_link and job_name and zone \
          and cluster and project):
    raise ValueError('Pubsub message must contain 7 required fields: '
                     'events_dir, test_name, logs_link, job_name, '
                     'zone, cluster, project. Message was: {}'.format(event))
  if not regression_test_config and not metric_collection_config:
    raise ValueError('metric_collection_config and regression_test_config '
                     'were both null; stopping early. See README for '
                     'documentation on writing these configs.')

  status, stop_time, num_failures = status_handler.get_job_status(
      job_name, job_namespace)
  if status == job_status_handler.UNKNOWN_STATUS:
    logger.warning(
        'Unknown status for job_name: {}. Message will be '
        'retried later.'.format(job_name))
    return False  # Do not ack the message.
  elif status == job_status_handler.DOES_NOT_EXIST:
    if msg_age_sec >= 60 * 60 * 24:
      logger.warning(
          'Job with job_name: {} no longer exists in Kubernetes. Message '
          'will be acknowledged.'.format(job_name))
      return True  # Ack the message.
    else:
      logger.warning(
          'Job with job_name: {} not found in Kubernetes. Message '
          'will be retried later.'.format(job_name))
      return False  # Do not ack the message.
  job_status = {
      'final_status': status,
      'start_time': publish_time,
      'publish_time': publish_time,
      'stop_time': stop_time,
      'num_failures': num_failures,
  }

  # TODO: pass these in the pubsub message and remove this block.
  if not test_type:
    test_type = 'func' if 'func' in test_name else 'conv'
  if not accelerator:
    accelerator = 'tpu-v2-8' if 'v2-8' in test_name else 'tpu-v3-8'
  if not framework_version:
    framework_version = 'pt-nightly' if 'pt-nightly' in test_name \
        else 'tf-nightly'

  handler = CloudMetricsHandler(
      test_name, events_dir, debug_info, metric_collection_config,
      regression_test_config, test_type, accelerator, framework_version, logger)

  # Sometimes pubsub messages get delayed. If we've already processed metrics
  # for a different attempt of this test, we need to see if that attempt came
  # before or after the current attempt.
  existing_row_uuid, existing_row_publish_time = handler.get_existing_row()
  if existing_row_publish_time:
    # If the current message is for an earlier attempt than the existing row,
    # we can stop early since we want to write metrics for the latest attempt.
    # Otherwise, proceed with processing the current message.
    if publish_time <= existing_row_publish_time:
      return True  # Ack the message.

  # Alert for failing jobs unless the user has explicitly added a config
  # that disables alerts for this test.
  if job_status['final_status'] != job_status_handler.SUCCESS and (
      not regression_test_config or regression_test_config.get(
          'alert_for_failed_jobs', True)):
    logger.error(
        'job_status was `{}` for test `{}`'.format(
            job_status['final_status'], test_name),
        debug_info=debug_info)

  new_metrics = handler.get_metrics_from_events_dir()
  handler.add_computed_metrics(new_metrics, job_status)
  if regression_test_config:
    metrics_history = handler.get_metrics_history_from_bigquery()
    metric_name_to_visual_bounds = handler.compute_bounds_and_report_errors(
        metrics_history, new_metrics, job_status['final_status'])
  else:
    metric_name_to_visual_bounds = None

  handler.add_status_and_metrics_to_bigquery(
      job_status, new_metrics, metric_name_to_visual_bounds)
  return True  # Ack the message.


def run_main(event, context):
  project_id = google.auth.default()[1]
  logger = alert_handler.AlertHandler(project_id)

  # Retrieve pubsub messages for all the tests that have been kicked off by
  # the test runner.
  subscriber = pubsub_v1.SubscriberClient()
  project = subscriber.project_path(project_id)
  subscription = None
  for s in subscriber.list_subscriptions(project):
    if s.topic.split('/')[-1] == METRICS_WRITTEN_TOPIC:
      subscription = s.name
      break
  if not subscription:
    subscription_id = subscriber.subscription_path(
        project_id, 'metrics-handler-subscription')
    topic = subscriber.topic_path(project_id, METRICS_WRITTEN_TOPIC)
    subscription = subscriber.create_subscription(
        subscription_id, topic, ack_deadline_seconds=300).name
  try:
    all_msgs = subscriber.pull(subscription, 100).received_messages
  except google.api_core.exceptions.DeadlineExceeded:
    logger.info(
        'No messages found for subscription: {}'.format(subscription))
    return

  # Group messages by test. Each test might have made multiple attempts and
  # therefore could have multiple messages.
  test_name_to_msgs = collections.defaultdict(list)
  ids_to_ack = []
  for msg in all_msgs:
    data_str = msg.message.data
    try:
      data = json.loads(data_str)
      data['publish_time'] = msg.message.publish_time.seconds
      data['ack_id'] = msg.ack_id
      test_name_to_msgs[data['test_name']].append(data)
    except Exception as e:
      logger.error(
          'Metrics handler encountered an invalid message in pubsub queue '
          'for topic `{}` which led to Exception: {}. This message will '
          'be acknowledged and ignored. The message was: {}'.format(
              METRICS_WRITTEN_TOPIC, e, msg))
      ids_to_ack.append(msg.ack_id)

  # Grab the latest message for each test. We will process only that message
  # and all other messages for that test will be ack'ed without being processed.
  msgs_to_process = []
  for test_name, msgs in test_name_to_msgs.items():
    sorted_msgs = sorted(msgs, key = lambda x: x['publish_time'])
    ids_to_ack.extend([msg['ack_id'] for msg in msgs[:-1]])
    msgs_to_process.append(msgs[-1])
  logger.info('Finished deduplicating messages from test runs.')

  # Note: it's good to ack early and often since pubsub will resend messages
  # that are not ack'ed within the queue's deadline.
  if ids_to_ack:
    logger.info('Will ack these ids: {}'.format(ids_to_ack))
    subscriber.acknowledge(subscription, ids_to_ack)
    logger.info('Successful ack for ids: {}'.format(ids_to_ack))

  if not msgs_to_process:
    logger.info('No messages to process. Stopping early.')
    return

  # TODO: Add support for multi-zone and/or multi-cluster setups.
  zone = msgs_to_process[0].get('zone')
  cluster = msgs_to_process[0].get('cluster_name')
  status_handler = job_status_handler.JobStatusHandler(
      project_id, zone, cluster, logger)

  # Handle the metrics for each test. Ack if the process was successful or if
  # the message is permanently invalid. Do not ack if the test is still running
  # so that we will retry again later once that test has finished running.
  for msg in msgs_to_process:
    try:
      logger.info('Pubsub message to process: {}'.format(msg))
      should_ack = _process_pubsub_message(msg, status_handler, logger)
    except Exception:
      logger.error(
          'Encountered exception while attempting to process message {}. '
          'The message will be acknowledged to prevent more crashes. '
          'Exception: {}'.format(msg, traceback.format_exc()))
      should_ack = True
    if should_ack:
      logger.info('Finished processing message. Will ack')
      subscriber.acknowledge(subscription, [msg['ack_id']])
      logger.info('Acknowledged ack_id: {}'.format(msg['ack_id']))
    else:
      logger.info('Finished processing message. Will not ack')
  logger.info('Processed a message for each of the following tests: '
              '{}'.format([x['test_name'] for x in msgs_to_process]))
  logger.send_email()
