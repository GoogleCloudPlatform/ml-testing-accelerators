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
import uuid

import alert_handler
import job_status_handler

import google.api_core.exceptions
from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.cloud import storage as gcs
import numpy as np
from tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf


ALLOWED_COMPARISONS = ['greater', 'less', 'equal', 'less_or_equal']
BQ_DATASET_NAME = 'metrics_handler_dataset'
BQ_JOB_TABLE_NAME = 'job_history'
BQ_METRIC_TABLE_NAME = 'metric_history'
METRICS_WRITTEN_TOPIC = 'metrics-written'
MIN_MSG_AGE_SEC = 60


class CloudMetricsHandler(object):
  def __init__(self, test_name, events_dir, stackdriver_logs_link,
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
      stackdriver_logs_link (string): Link to the Stackdriver Logs for the run
        that produced the metrics being handled.
      metric_collection_config (dict): Options for collecting metrics. See
        README for documentation.
      regression_test_config (dict): Options for alerting in the event of
        metrics regressions. See README for documentation.
      test_type (string): E.g. 'convergence' or 'functional'. Used to organize
        metrics in Bigquery.
      accelerator (string): E.g. 'tpu-v2-8'. The type of accelerator used to
        run this test. Used to organize metrics in Bigquery.
      framework_version (string): E.g. 'pt-nightly' or 'tf-nightly'. The
        combined ML framework + version identifier used to run this test. Used
        to organize metrics in Bigquery.
      logger (`AlertHandler` instance): Used to write logs and alert emails.
    """
    self.MetricPoint = collections.namedtuple(
        'MetricPoint', ['metric_value', 'wall_time'])
    self.test_name = test_name

    self.events_dir = events_dir
    self.stackdriver_logs_link = stackdriver_logs_link
    self.metric_collection_config = metric_collection_config or {}
    self.regression_test_config = regression_test_config or {}
    self.test_type = test_type
    self.accelerator = accelerator
    self.framework_version = framework_version
    self.logger = logger

    # Initalize clients to interact with various Cloud APIs.
    self.project = google.auth.default()[1]
    self.bigquery_client = bigquery.Client()
    self.gcs_client = gcs.Client()

    self.job_history_table_id = self._get_table_id(
        BQ_DATASET_NAME, BQ_JOB_TABLE_NAME)
    self.metric_history_table_id = self._get_table_id(
        BQ_DATASET_NAME, BQ_METRIC_TABLE_NAME)
    if self.metric_collection_config.get('write_to_bigquery'):
      self._make_bigquery_tables()


  @staticmethod
  def _wall_time_to_sql_timestamp(wall_time):
    return datetime.datetime.fromtimestamp(wall_time).strftime(
        '%Y-%m-%d %H:%M:%S')


  def _get_table_id(self, dataset_name, table_name):
     return '{}.{}.{}'.format(self.project, dataset_name, table_name)


  def add_point_to_metrics(self, metrics, metric_name, metric_value, wall_time):
    """Adds a metric data point to `metrics`.

    Args:
      metrics (dict): Key is metric name is value is a MetricPoint.
      metric_name (string): Name of the metric being added.
      metric_value (*): Value to assign to this metric.
      wall_time (int): Seconds since epoch.

    Raises:
      ValueError if `metric_name` is already in `metrics`.
    """
    if metric_name in metrics:
      raise ValueError('{} was already in metrics with value {}.'.format(
          metric_name, metrics[metric_name]))
    metrics[metric_name] = self.MetricPoint(
        metric_value=metric_value,
        wall_time=wall_time)


  def _add_time_to_accuracy_to_metrics(self, raw_metrics, metrics_to_update):
    tta_config = self.metric_collection_config['time_to_accuracy']
    if 'accuracy_tag' not in tta_config or \
        'accuracy_threshold' not in tta_config:
      raise ValueError('Invalid `time_to_accuracy` portion of config. '
                       'See README for how to set up the config.')
    values = raw_metrics.get(tta_config['accuracy_tag'], [])
    if not values:
      raise ValueError('No values found for time to accuracy tag: {}. '
          'Possible tags were: {}'.format(
          tta_config['accuracy_tag'], raw_metrics.keys()))

    # MetricPoints should be sorted by timestamp with earlier events first.
    start_wall_time = values[0].wall_time
    try:
      end_wall_time = next(
          v.wall_time for v in values
          if v.metric_value >= tta_config['accuracy_threshold'])
      self.add_point_to_metrics(
          metrics_to_update, 'time_to_accuracy',
          (end_wall_time - start_wall_time), end_wall_time)
    except StopIteration:
      max_accuracy = max(v.metric_value for v in values)
      self.logger.error(
          'Accuracy was never high enough to satisfy the `time_to_accuracy` '
          'settings from the config. Max accuracy: {}. Config for '
          '`time_to_accuracy`: {}'.format(
              max_accuracy, tta_config),
          logs_link=self.stackdriver_logs_link)


  def _add_total_wall_time_to_metrics(self, raw_metrics, metrics_to_update):
    if not raw_metrics:
      self.logger.warning('Empty raw_metrics; skipping total_wall_time')
      return
    values = list(itertools.chain.from_iterable(raw_metrics.values()))
    min_wall_time = min(v.wall_time for v in values)
    max_wall_time = max(v.wall_time for v in values)
    self.add_point_to_metrics(
        metrics_to_update, 'total_wall_time', (max_wall_time - min_wall_time),
        max_wall_time)


  def _aggregate_metrics(self, metrics, strategy):
    if strategy == 'final':
      # Take the MetricPoint with the latest wall_time.
      latest_metric = metrics[0]
      for metric in metrics[1:]:
        if metric.wall_time > latest_metric.wall_time:
          latest_metric = metric
      return latest_metric
    elif strategy == 'max':
      # Take the MetricPoint with the maximum metric value.
      max_metric = metrics[0]
      for metric in metrics[1:]:
        if metric.metric_value > max_metric.metric_value:
          max_metric = metric
      return max_metric
    elif strategy == 'min':
      # Take the MetricPoint with the minimum metric value.
      min_metric = metrics[0]
      for metric in metrics[1:]:
        if metric.metric_value < min_metric.metric_value:
          min_metric = metric
      return min_metric
    else:
      raise ValueError('Unknown aggregation strategy: {}'.format(strategy))


  def get_metrics_from_events_dir(self):
    """Retrieves and aggregates metrics from Tensorboard Summary file.

    Returns:
      final_metrics(dict): Key is metric name and value is a MetricPoint
        containing the aggregated value for that metric.
    """
    tags_to_ignore = set(
        self.metric_collection_config.get('tags_to_ignore', []))

    em = event_multiplexer.EventMultiplexer()
    em.AddRunsFromDirectory(self.events_dir)
    em.Reload()

    # First pass: collect the values for each metric.
    raw_metrics = collections.defaultdict(list)
    for run, tags in em.Runs().items():
      # 'Old-style' runs have a simple format and store values directly.
      for tag in tags['scalars']:
        if tag in tags_to_ignore:
          continue
        raw_metrics[tag].extend(
            [self.MetricPoint(metric_value=x.value, wall_time=x.wall_time)
            for x in em.Scalars(run, tag)])
      # 'New-style' runs stores values inside of Tensor protos.
      for tag in tags['tensors']:
        if tag in tags_to_ignore:
          continue
        for t in em.Tensors(run, tag):
          tensor_dtype = tf.dtypes.as_dtype(t.tensor_proto.dtype)
          try:
            val = np.frombuffer(
                t.tensor_proto.tensor_content,
                tensor_dtype.as_numpy_dtype).tolist()
            assert len(val) == 1  # There should be 1 value per tensor.
            raw_metrics[tag].append(
                self.MetricPoint(metric_value=val[0], wall_time=t.wall_time))
          except ValueError as e:
            self.logger.warning(
                'Unable to parse tag: `{}` from tensor_content: {}. '
                'Error: {}. Consider adding this tag to tags_to_ignore '
                'in config.'.format(tag, t.tensor_proto.tensor_content, e))

    # Second pass: aggregate values for each metric based on the config.
    final_metrics = {}
    tag_to_custom_aggregation_strategies = self.metric_collection_config.get(
        'metric_to_aggregation_strategies', {})
    for tag, metrics in raw_metrics.items():
      strategies = tag_to_custom_aggregation_strategies.get(
          tag, self.metric_collection_config['default_aggregation_strategies'])
      for strategy in strategies:
        final_metrics['{}_{}'.format(tag, strategy)] = self._aggregate_metrics(
            metrics, strategy)

    # Compute time_to_accuracy if requested in the config.
    if 'time_to_accuracy' in self.metric_collection_config:
      self._add_time_to_accuracy_to_metrics(raw_metrics, final_metrics)

    self._add_total_wall_time_to_metrics(raw_metrics, final_metrics)
    return final_metrics


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
      metric_name_to_visual_bounds={}):
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
        self.stackdriver_logs_link,
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
      errors = self.bigquery_client.insert_rows(table, rows)
      if errors == []:
        self.logger.info('Successfully added rows to Bigquery.')
      else:
        # TODO: Maybe add retry logic. insert_rows seems to be atomic for all
        #       elements in the list, so it should be safe to retry.
        self.logger.error(
            'Failed to add rows to Bigquery. Errors: {}'.format(errors),
            logs_link=self.stackdriver_logs_link)


  def _get_metrics_history_from_bigquery(self):
    query_result = self.bigquery_client.query(
        'SELECT * FROM `{}` WHERE test_name like \"{}\"'.format(
            self.metric_history_table_id, self.test_name)).result()
    metrics_history = collections.defaultdict(list)
    for row in query_result:
      metrics_history[row['metric_name']].append(self.MetricPoint(
          metric_value=row['metric_value'],
          wall_time=row['timestamp'].timestamp()))
    return metrics_history


  def _metric_bounds(self, metric_name, value_history):
    """Compute upper/lower bounds and whether metric is within those bounds.

    Args:
      metric_name (string): Name of the metric to check.
      value_history (list of floats): History of values for this metric. These
        should be ordered so the most recent value is the last in the list.

    Returns:
      tuple(is_within_bounds (bool), lower_bound (float), upper_bound (float)).
        lower_bound and/or upper_bound can be None.

    Raises:
      ValueError if the regression test config is invalid.
    """
    success_conditions = self.regression_test_config.get(
        'metric_success_conditions', {})
    success_condition = success_conditions.get(metric_name) or \
        success_conditions.get('default')
    if not success_condition:
      self.logger.warning(
          'metric: `{}` has an empty success condition in metric_opt_in_dict '
          'but there is no default condition provided in the regression '
          'test config. No bounds or alerts will be computed'.format(
              metric_name))
      return True, None, None
    if len(value_history) <= success_condition.get(
        'wait_for_n_points_of_history', -1):
      self.logger.info(
          'Metric: {} had only {} points of history. Skipping bounds '
          'enforcement. Success condition: {}'.format(
              metric_name, len(value_history), success_condition))
      return True, None, None

    metric_value = value_history[-1]
    comparison = success_condition.get('comparison')
    if not comparison or comparison not in ALLOWED_COMPARISONS:
      raise ValueError(
          'A metric success condition must set the `comparison` field to '
          'one of {}. Condition was: {}'.format(
              ALLOWED_COMPARISONS, success_condition))
    thresholds = [x for x in success_condition['success_threshold'].items()]
    if len(thresholds) != 1:
      raise ValueError('Each metric success condition should have exactly '
                       '1 success_threshold. Condition was: {}'.format(
                           success_condition))

    threshold_type, threshold_value = thresholds[0]

    if threshold_type == 'fixed_value':
      if comparison == 'greater':
        visual_lower_bound = threshold_value
        visual_upper_bound = None
        within_bounds = metric_value > threshold_value
      elif comparison == 'less':
        visual_lower_bound = None
        visual_upper_bound = threshold_value
        within_bounds = metric_value < threshold_value
      elif comparison == 'equal':
        visual_lower_bound = threshold_value
        visual_upper_bound = threshold_value
        within_bounds = math.isclose(metric_value, threshold_value)
      elif comparison == 'less_or_equal':
        visual_lower_bound = None
        visual_upper_bound = threshold_value
        within_bounds = math.isclose(metric_value, threshold_value) or \
            metric_value < threshold_value
      return within_bounds, visual_lower_bound, visual_upper_bound

    if threshold_type == 'stddevs_from_mean':
      v_mean = np.mean(value_history)
      v_stddev = np.std(value_history)
      visual_lower_bound = max(v_mean - (v_stddev * threshold_value), 0)
      visual_upper_bound = v_mean + (v_stddev * threshold_value)
      if comparison == 'greater':
        within_bounds = metric_value > visual_lower_bound
      elif comparison == 'less':
        within_bounds = metric_value < visual_upper_bound
      elif comparison == 'less_or_equal':
        within_bounds = math.isclose(metric_value, visual_upper_bound) or \
            metric_value < visual_upper_bound
      else:
        raise ValueError(
            'A metric success condition using a `stddevs_from_mean`-type '
            'threshold must use `greater`, `less`, or `less_or_equal` for the '
            'comparison. The condition was: {}'.format(success_condition))
      return within_bounds, visual_lower_bound, visual_upper_bound

    raise ValueError(
        'The threshold type of a metric success condition should be either '
        '`fixed_value` or `stddevs_from_mean`. Condition was: {}'.format(
            success_condition))


  def compute_bounds_and_report_errors(self, job_status, new_metrics):
    """Compute the bounds for metrics and report abnormal values.

    Any metric that is currently outside the expected bounds is reported to
    Stackdriver Error Reporting unless `write_to_error_reporting` is set to
    False in the regression test config. Even if this reporting is turned off,
    this method computes the upper and lower bounds for each metric to provide
    to BigQuery as a visual aid when rendering metrics history into charts.

    Args:
      job_status (dict): Contains information about the Kubernetes Job.
      new_metrics(dict): Key is metric name and value is MetricPoint containing
        the latest aggregated value for that metric.

    Returns:
      metric_name_to_visual_bounds (dict): Key is metric name and value is a
        tuple of floats of the form (lower_bound, upper_bound).
    """
    metrics_history = self._get_metrics_history_from_bigquery()

    # Add the metrics from the latest run. These aren't in Bigquery yet.
    for metric_name, metric_value in new_metrics.items():
      metrics_history[metric_name].append(metric_value)

    if job_status['final_status'] != job_status_handler.SUCCESS:
      self.logger.error(
          'job_status was `{}` for test `{}`'.format(
              job_status['final_status'], self.test_name),
          logs_link=self.stackdriver_logs_link)

    metric_name_to_visual_bounds = {}
    metric_subset_to_report = set(
        self.regression_test_config.get('metric_subset_to_alert', []))
    for metric_name, metric_value_list in metrics_history.items():
      if metric_subset_to_report and metric_name not in metric_subset_to_report:
        self.logger.info(
            'Skipping alerts and bounds for metric `{}` since '
            'it does not appear in `metric_subset_to_report` in your '
            'regression test config.'.format(metric_name))
        continue
      value_history = [v.metric_value for v in metric_value_list]
      within_bounds, lower_bound, upper_bound = self._metric_bounds(
          metric_name, value_history)
      if lower_bound is not None or upper_bound is not None:
        metric_name_to_visual_bounds[metric_name] = (lower_bound, upper_bound)

      # Report out-of-bounds metrics to Stackdriver unless disabled by config.
      if within_bounds or not self.regression_test_config.get(
          'write_to_error_reporting', True):
        continue
      lower_bound = '-inf' if lower_bound is None else '{:.2f}'.format(
          lower_bound)
      upper_bound = 'inf' if upper_bound is None else '{:.2f}'.format(
          upper_bound)
      self.logger.error(
          'Metric `{}` was out of bounds for test `{}`. Bounds were '
          '({}, {}) and value was {}'.format(
              metric_name, self.test_name, lower_bound, upper_bound,
              '{:.2f}'.format(value_history[-1])),
          logs_link=self.stackdriver_logs_link)

    return metric_name_to_visual_bounds


def _process_pubsub_message(msg, status_handler, logger):
  msg_age_sec = time.time() - msg['publish_time']
  if msg_age_sec < MIN_MSG_AGE_SEC:
    logger.warning('Message was {} seconds old, which is less than the '
                   'minimum of {}. Skipping for now but will retry on '
                   'the next run.'.format(msg_age_sec, MIN_MSG_AGE_SEC))
    return False  # Do not ack the message.
  events_dir = msg.get('model_dir')
  test_name = msg.get('test_name')
  logs_link = msg.get('logs_link')
  metric_collection_config = msg.get('metric_collection_config')
  regression_test_config = msg.get('regression_test_config')

  job_name = msg.get('job_name')
  job_namespace = msg.get('job_namespace')
  test_type = msg.get('test_type')
  accelerator = msg.get('accelerator')
  framework_version = msg.get('framework_version')

  if not (events_dir and test_name and logs_link and job_name):
    raise ValueError('Pubsub message must contain 4 required fields: '
                     'events_dir, test_name, logs_link, and job_name. '
                     'Message was: {}'.format(event))
  if not regression_test_config and not metric_collection_config:
    raise ValueError('metric_collection_config and regression_test_config '
                     'were both null; stopping early. See README for '
                     'documentation on writing these configs.')

  status, start_time, stop_time, num_failures = status_handler.get_job_status(
      job_name, job_namespace)
  if status == job_status_handler.UNKNOWN_STATUS:
    logger.warning(
        'Unknown status for job_name: {}. Message will be '
        'retried later.'.format(job_name))
    return False  # Do not ack the message.
  elif status == job_status_handler.DOES_NOT_EXIST:
    logger.warning(
        'Job with job_name: {} no longer exists in Kubernetes. Message '
        'will be acknowledged.'.format(job_name))
    return True  # Ack the message.
  job_status = {
      'final_status': status,
      'start_time': start_time,
      'stop_time': stop_time,
      'num_failures': num_failures,
  }

  # TODO: pass these in the pubsub message and remove this block.
  if not test_type:
    test_type = 'functional' if 'functional' in test_name else 'convergence'
  if not accelerator:
    accelerator = 'tpu-v2-8' if 'v2-8' in test_name else 'tpu-v3-8'
  if not framework_version:
    framework_version = 'pt-nightly' if 'pt-nightly' in test_name \
        else 'tf-nightly'

  handler = CloudMetricsHandler(
      test_name, events_dir, logs_link, metric_collection_config,
      regression_test_config, test_type, accelerator, framework_version, logger)

  new_metrics = handler.get_metrics_from_events_dir()
  metric_name_to_visual_bounds = handler.compute_bounds_and_report_errors(
      job_status, new_metrics)
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
    except Exception as e:
      logger.error(
          'Encountered exception `{}` while attempting to '
          'process message. The message will be acknowledged to prevent more '
          'crashes. The message was: {}'.format(e, msg))
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

run_main(None, None)
