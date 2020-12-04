import dataclasses
import datetime
import math
import typing

from absl import logging
from google.cloud import bigquery

BQ_JOB_TABLE_NAME = 'job_history'
BQ_METRIC_TABLE_NAME = 'metric_history'

@dataclasses.dataclass
class JobHistoryRow:
  uuid: str
  test_name: str
  test_type: str
  accelerator: str
  framework_version: str
  job_status: str
  num_failures: int
  job_duration_sec: int
  timestamp: datetime.datetime
  stackdriver_logs_link: str
  msg_publish_time: typing.Optional[str] = None
  logs_download_command: typing.Optional[str] = None
  kubernetes_workload_link: typing.Optional[str] = None

@dataclasses.dataclass
class MetricHistoryRow:
  uuid: str
  test_name: str
  timestamp: datetime.datetime
  metric_name: str
  metric_value: float
  metric_lower_bound: typing.Optional[float] = None
  metric_upper_bound: typing.Optional[float] = None

def _to_bigquery_schema(dataclass):
  python_type_to_bq_type = {
    str: ("STRING", "REQUIRED"),
    int: ("INT64", "REQUIRED"),
    float: ("FLOAT64", "REQUIRED"),
    datetime.datetime: ("TIMESTAMP", "REQUIRED"),
  }
  # Add Optional types to dict
  python_type_to_bq_type.update({
    typing.Optional[pt]: (bqt, "NULLABLE")
    for pt, (bqt, _) in python_type_to_bq_type.items()
  })

  return [
    bigquery.SchemaField(field.name, *python_type_to_bq_type[field.type])
    for field in dataclasses.fields(dataclass)
  ]

def _is_valid_value(v):
  """Return True if value is valid for writing to BigQuery.

  Args:
    v (anything): Value to check.

  Returns:
    Bool, True if v is valid and False otherwise.

  """
  invalid_values = [math.inf, -math.inf, math.nan]
  if v in invalid_values:
    return False
  try:
    if math.isnan(v):
      return False
  except TypeError:
    pass
  return True

def _replace_invalid_values(row):
  """Replace float values that are not available in BigQuery.

  Args:
    row: List of values to insert into BigQuery.

  Returns:
    List, `row` with invalid values replaced with `None`.
  """
  return [x if _is_valid_value(x) else None for x in row]

class BigQueryMetricStore:
  def __init__(self, dataset: str, project: typing.Optional[str] = None):
    self._dataset = dataset
    self._project = project or google.auth.default()[1]
    self.bigquery_client = bigquery.Client(
        project=project,
        default_query_job_config=bigquery.job.QueryJobConfig(
          default_dataset=".".join((self._project, self._dataset)),
        )
    )

  @property
  def job_history_table_id(self):
    return ".".join((self._project, self._dataset, BQ_JOB_TABLE_NAME))

  @property
  def metric_history_table_id(self):
    return ".".join((self._project, self._dataset, BQ_METRIC_TABLE_NAME))

  def create_tables(self):
    dataset = bigquery.Dataset(self.bigquery_client.dataset(self._dataset))
    _ = self.bigquery_client.create_dataset(dataset, exists_ok=True)

    job_history_schema = _to_bigquery_schema(JobHistoryRow)
    job_history_table = bigquery.Table(
        self.job_history_table_id, schema=job_history_schema)
    _ = self.bigquery_client.create_table(job_history_table, exists_ok=True)

    metric_history_schema = _to_bigquery_schema(MetricHistoryRow)
    metric_history_table = bigquery.Table(
        self.metric_history_table_id, schema=metric_history_schema)
    _ = self.bigquery_client.create_table(metric_history_table, exists_ok=True)

  def insert_status_and_metrics(
      self, 
      job: JobHistoryRow,
      metrics: typing.Iterable[MetricHistoryRow]):

    # Every job should have 1 job status row and it should exist even if
    # no other metrics exist.
    job_history_rows = [dataclasses.astuple(job)]

    # Create rows to represent the computed metrics for this job.
    metric_history_rows = []
    for metric in metrics:
      if not _is_valid_value(float(metric.metric_value)):
        logging.warning(
            'Found metric row with invalid value: {} {} {}'.format(
                job.test_name,
                metric.metric_name,
                metric.metric_value))
        continue

      metric_history_rows.append(dataclasses.astuple(metric))

    # Insert rows in Bigquery.
    for table_id, rows in [
        (self.job_history_table_id, job_history_rows),
        (self.metric_history_table_id, metric_history_rows),
    ]:
      if not rows:
        continue
      logging.info(
          'Inserting {} rows into BigQuery table `{}`'.format(
              len(rows), table_id))
      table = self.bigquery_client.get_table(table_id)
      clean_rows = [_replace_invalid_values(row) for row in rows]
      errors = self.bigquery_client.insert_rows(table, clean_rows)
      if not errors:
        logging.info('Successfully added rows to Bigquery.')
      else:
        logging.error(
            'Failed to add rows to Bigquery. Errors: {}'.format(errors))

  def get_metric_history(
      self,
      benchmark_id: str,
      metric_key: str,
      min_time: typing.Optional[datetime.datetime] = None
  ) -> typing.Iterable[MetricHistoryRow]:
    """Returns the historic values of each metric for a given model."""
    query = """
        SELECT *
        FROM `metric_history`
        WHERE test_name LIKE @benchmark_id AND
          metric_name LIKE @metric_key AND
          (metric_lower_bound IS NULL OR metric_value >= metric_lower_bound) AND
          (metric_upper_bound IS NULL OR metric_value <= metric_upper_bound) AND
          timestamp >= @min_time AND
          uuid IN (
              SELECT uuid
              FROM `job_history`
              WHERE test_name LIKE @benchmark_id AND job_status = \"success"\
          )
    """
    job_config = bigquery.QueryJobConfig(
      query_parameters =[
        bigquery.ScalarQueryParameter("benchmark_id",
            "STRING", benchmark_id),
        bigquery.ScalarQueryParameter("metric_key",
            "STRING", metric_key),
        bigquery.ScalarQueryParameter("min_time",
            "TIMESTAMP", min_time),
      ]
    )
    query_result = self.bigquery_client.query(
        query, job_config=job_config)
    
    return [MetricHistoryRow(**row) for row in query_result]

'''
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
'''
