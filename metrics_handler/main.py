import base64
import collections
import datetime
import io
import itertools
import json
import math
import time

import job_status_handler

from absl import logging
import google.api_core.exceptions
from google.cloud import bigquery
from google.cloud import error_reporting
from google.cloud import pubsub_v1
from google.cloud import storage as gcs
import numpy as np
from tensorboard.backend.event_processing import event_multiplexer
import tensorflow as tf


ERROR_REPORTER = error_reporting.Client()
METRICS_WRITTEN_TOPIC = 'metrics-written'
MIN_MSG_AGE_SEC = 30


class CloudMetricsHandler(object):
  def __init__(self, test_name, events_dir, stackdriver_logs_link,
               metric_collection_config, regression_test_config,
               job_name):
    """Handles metrics storage, collection, aggregation, alerts, etc.

    Used in conjunction with the Cloud Accelerator Testing framework
    (TODO: link to top-level Github repo for the framework).

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
      job_name (string): Name of the Kubernetes Job that ran this instance
        of the test.
    """
    self.MetricPoint = collections.namedtuple(
        'MetricPoint', ['metric_value', 'wall_time'])
    self.test_name = test_name
    self.events_dir = events_dir
    self.stackdriver_logs_link = stackdriver_logs_link
    self.job_name = job_name
    self.metric_collection_config = metric_collection_config or {}
    self.regression_test_config = regression_test_config or {}

    # Initalize clients to interact with various Cloud APIs.
    self.project = google.auth.default()[1]
    self.bigquery_client = bigquery.Client()
    self.gcs_client = gcs.Client()


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
      ERROR_REPORTER.report(
          'Accuracy was never high enough to satisfy the `time_to_accuracy` '
          'settings from the config. Max accuracy: {}. Config for '
          '`time_to_accuracy`: {}. Logs for this run: {}'.format(
              max_accuracy, tta_config, self.stackdriver_logs_link))


  def _add_total_wall_time_to_metrics(self, raw_metrics, metrics_to_update):
    if not raw_metrics:
      logging.warning('Empty raw_metrics; skipping total_wall_time')
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
            logging.error(
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


  def _make_bigquery_table(self):
    if not self.metric_collection_config.get('write_to_bigquery'):
      return
    dataset_name = self.metric_collection_config['bigquery_dataset_name']
    dataset = bigquery.Dataset(self.bigquery_client.dataset(dataset_name))
    _ = self.bigquery_client.create_dataset(dataset, exists_ok=True)

    table_id = self._get_table_id(
        dataset_name, self.metric_collection_config['bigquery_table_name'])
    schema = [
        bigquery.SchemaField("test_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_value", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("stackdriver_logs_link", "STRING",
                             mode="REQUIRED"),
        bigquery.SchemaField("metric_upper_bound", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("metric_lower_bound", "FLOAT64", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    _ = self.bigquery_client.create_table(table, exists_ok=True)
    return table_id


  def add_new_metrics_to_bigquery(self, aggregated_metrics_dict,
                                  metric_name_to_visual_bounds={}):
    """Adds rows to the BigQuery that contains metrics history.

    Args:
      aggregated_metrics_dict (dict): Key is metric name and value is a
        MetricPoint containing the aggregated value for that metric.
      metric_name_to_visual_bounds (dict, optional): Key is metric name and
        value is a tuple consisting of (lower_bound, upper bound) for that
        metric. These bounds are useful for showing helpful lines on charts
        of metrics history. If not provided, the rows saved to BigQuery will
        have `null` for upper_bound and lower_bound.
    """
    if not self.metric_collection_config.get('write_to_bigquery', False):
      logging.info('Skipping writing metrics to BigQuery.')
      return
    if not aggregated_metrics_dict:
      logging.warning('No metrics to write to BigQuery.')
      return
    table_id = self._make_bigquery_table()
    rows_to_insert = []
    for metric_name, metric_point in aggregated_metrics_dict.items():
      lower_bound, upper_bound = metric_name_to_visual_bounds.get(
          metric_name, (None, None))
      rows_to_insert.append([
        self.test_name,
        metric_name,
        float(metric_point.metric_value),
        self._wall_time_to_sql_timestamp(metric_point.wall_time),
        self.stackdriver_logs_link,
        upper_bound,
        lower_bound,
      ])
    logging.info('Inserting {} rows into BigQuery table `{}`'.format(
        len(rows_to_insert), table_id))
    table = self.bigquery_client.get_table(table_id)
    errors = self.bigquery_client.insert_rows(table, rows_to_insert)
    if errors == []:
      logging.info('Successfully added rows to Bigquery.')
    else:
      # TODO: Maybe add retry logic. insert_rows seems to be atomic for all
      #       elements in the list, so it should be safe to retry.
      logging.error(
          'Failed to add rows to Bigquery. Errors: {}'.format(errors))


  def _get_metrics_history_from_bigquery(self):
    dataset_name = self.regression_test_config.get(
        'bigquery_dataset_name',
        self.metric_collection_config.get('bigquery_dataset_name'))
    table_name = self.regression_test_config.get(
        'bigquery_table_name',
        self.metric_collection_config.get('bigquery_table_name'))
    table_id = self._get_table_id(dataset_name, table_name)
    query_result = self.bigquery_client.query(
        'SELECT * FROM `{}` WHERE test_name like \"{}\"'.format(
            table_id, self.test_name)).result()
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
    success_condition = self.regression_test_config.get(
        'metric_success_conditions', {}).get(metric_name) or \
        self.regression_test_config.get('default_metric_success_condition')
    if not success_condition:
      logging.warning(
          'metric: `{}` has an empty success condition in metric_opt_in_dict '
          'but there is no default_metric_success_condition set in regression '
          'test config. No bounds or alerts will be computed'.format(
              metric_name))
      return True, None, None
    if 0 and len(value_history) <= success_condition.get(
        'wait_for_n_points_of_history', -1):
      logging.info(
          'Metric: {} had only {} points of history. Skipping bounds '
          'enforcement. Success condition: {}'.format(
              metric_name, len(value_history), success_condition))
      return True, None, None

    metric_value = value_history[-1]
    comparison = success_condition.get('comparison')
    if not comparison or comparison not in ['greater', 'less', 'equal']:
      raise ValueError(
          'A metric success condition must set the `comparison` field to '
          '`greater`, `less`, or `equal`. Condition was: {}'.format(
              success_condition))
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
      else:
        raise ValueError(
            'A metric success condition using a `stddevs_from_mean`-type '
            'threshold must use `greater` or `less` for the comparison. The '
            'condition was: {}'.format(success_condition))
      return within_bounds, visual_lower_bound, visual_upper_bound

    raise ValueError(
        'The threshold type of a metric success condition should be either '
        '`fixed_value` or `stddevs_from_mean`. Condition was: {}'.format(
            success_condition))


  def compute_bounds_and_report_errors(self, new_metrics):
    """Compute the bounds for metrics and report abnormal values.

    Any metric that is currently outside the expected bounds is reported to
    Stackdriver Error Reporting unless `write_to_error_reporting` is set to
    False in the regression test config. Even if this reporting is turned off,
    this method computes the upper and lower bounds for each metric to provide
    to BigQuery as a visual aid when rendering metrics history into charts.

    Args:
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

    metric_name_to_visual_bounds = {}
    metric_subset_to_report = set(
        self.regression_test_config.get('metric_subset_to_alert', []))
    for metric_name, metric_value_list in metrics_history.items():
      if metric_subset_to_report and metric_name not in metric_subset_to_report:
        logging.info(
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
      ERROR_REPORTER.report(
          'Metric `{}` was out of bounds for test `{}`. Bounds were '
          '({}, {}) and value was {}. Logs for this run: {}'.format(
              metric_name, self.test_name, lower_bound, upper_bound,
              '{:.2f}'.format(value_history[-1]),
              self.stackdriver_logs_link))

    return metric_name_to_visual_bounds


def _process_pubsub_message(msg, status_handler):
  msg_age_sec = time.time() - msg['publish_time']
  if msg_age_sec < MIN_MSG_AGE_SEC:
    logging.warning('Message was {} seconds old, which is less than the '
                    'minimum of {}. Skipping for now but will retry on '
                    'the next run.'.format(msg_age_sec, MIN_MSG_AGE_SEC))
    return False  # Do not ack the message.
  events_dir = msg.get('model_dir')
  test_name = msg.get('test_name')
  logs_link = msg.get('logs_link')
  metric_collection_config = msg.get('metric_collection_config')
  regression_test_config = msg.get('regression_test_config')
  job_name = msg.get('job_name')

  if not regression_test_config and not metric_collection_config:
    raise ValueError('metric_collection_config and regression_test_config '
                     'were both null; stopping early. See README for '
                     'documentation on writing these configs.')
  if not (events_dir and test_name and logs_link and job_name):
    raise ValueError('Pubsub message must contain 4 required fields: '
                     'events_dir, test_name, logs_link, and job_name. See '
                     'README for documentation. Message was: {}'.format(event))

  status_code, completion_time = status_handler.get_job_status(
      job_name, 'automated')
  if status_code == job_status_handler.UNKNOWN_STATUS_CODE:
      logging.warning(
          'Unknown status for job_name: {}. Message will be '
          'retried later.'.format(job_name))
      return False  # Do not ack the message.

  handler = CloudMetricsHandler(test_name, events_dir, logs_link,
      metric_collection_config, regression_test_config, job_name)

  new_metrics = handler.get_metrics_from_events_dir()
  handler.add_point_to_metrics(new_metrics, 'job_status', status_code,
                               completion_time)

  metric_name_to_visual_bounds = handler.compute_bounds_and_report_errors(
      new_metrics)
  handler.add_new_metrics_to_bigquery(
      new_metrics, metric_name_to_visual_bounds)
  return True  # Ack the message.


def run_main(event, context):
  project_id = google.auth.default()[1]

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
    logging.info('No messages found for subscription: {}'.format(subscription))
    exit(0)

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
      ERROR_REPORTER.report(
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
  logging.info('Finished deduplicating messages from test runs. ')

  # Note: it's good to ack early and often since pubsub will resend messages
  # that are not ack'ed within the queue's deadline.
  if ids_to_ack:
    logging.info('Will ack these ids: {}'.format(ids_to_ack))
    subscriber.acknowledge(subscription, ids_to_ack)
    logging.info('Successful ack for ids: {}'.format(ids_to_ack))

  # TODO: Consider adding these to the pubsub message and passing them in.
  zone = 'us-central1-b'
  cluster_id = 'xl-ml-test'
  status_handler = job_status_handler.JobStatusHandler(
      project_id, zone, cluster_id)

  # Handle the metrics for each test. Ack if the process was successful or if
  # the message is permanently invalid. Do not ack if the test is still running
  # so that we will retry again later once that test has finished running.
  for msg in msgs_to_process:
    try:
      logging.info('Pubsub message to process: {}'.format(msg))
      should_ack = _process_pubsub_message(msg, status_handler)
    except Exception as e:
      ERROR_REPORTER.report(
          'Encountered exception `{}` while attempting to process message. '
          'The message will be acknowledged to prevent more crashes. The '
          'message was: {}'.format(e, msg))
      should_ack = True
    if should_ack:
      logging.info('Finished processing message. Will ack')
      subscriber.acknowledge(subscription, [msg['ack_id']])
      logging.info('Acknowledged ack_id: {}'.format(msg['ack_id']))
    else:
      logging.info('Finished processing message. Will not ack')
  logging.info('Processed a message for each of the following tests: '
               '{}'.format([x['test_name'] for x in msgs_to_process]))
