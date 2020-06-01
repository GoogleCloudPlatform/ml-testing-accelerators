# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import math
import os

from absl import logging
from bokeh.layouts import column, row
from bokeh.models import CategoricalColorMapper, ColumnDataSource, Div, HoverTool
from bokeh.plotting import figure

import javascript_utils
import utils

import numpy as np

# For a given list of test_names and metric_names, find the history of metrics.
QUERY = """
SELECT
  metrics.test_name,
  metrics.metric_name,
  SAFE_CAST(DATE(metrics.timestamp, 'US/Pacific') AS STRING) AS run_date,
  metrics.metric_value,
  job.job_status,
  job.stackdriver_logs_link AS logs_link,
  job.logs_download_command,
  job.uuid
FROM (
  SELECT
    x.test_name,
    x.metric_name,
    x.timestamp,
    x.metric_value,
    x.uuid
  FROM (
    SELECT
      test_name,
      metric_name,
      SAFE_CAST(DATE(timestamp, 'US/Pacific') AS STRING) AS run_date,
      max(farm_fingerprint(uuid)) as max_uuid
    FROM
      `{metric_table_name}`
    WHERE
      timestamp > '{cutoff_date}' AND
      {test_name_where_clause} AND
      {metric_name_where_clause}
    GROUP BY
      test_name, metric_name, run_date
  ) AS y
  INNER JOIN `{metric_table_name}` AS x
  ON
    y.test_name = x.test_name AND
    y.metric_name = x.metric_name AND
    y.max_uuid = farm_fingerprint(x.uuid)
) AS metrics
INNER JOIN `{job_table_name}` AS job
ON
  metrics.uuid = job.uuid
ORDER BY
  run_date DESC
"""

def get_query_config(test_names, metric_names):
  query_params = []
  def _add_params(column_name, names):
    for i, name in enumerate(names):
      query_params.append({
        'name': f'{column_name}{i}',
        'parameterType': {'type': 'STRING'},
        'parameterValue': {'value': name},
      })
  _add_params('test_name', test_names)
  _add_params('metric_name', metric_names)
  return {
    'query': {
      'parameterMode': 'NAMED',
      'queryParameters': query_params,
    }
  }

def get_query(test_names, metric_names):
  # Note that Bigquery does not support ANY, otherwise we could use a simpler
  # query such as "WHERE test_name LIKE ANY(...)".
  def _make_where_clause(column_name, names):
    where_clause = f'({column_name} LIKE @{column_name}0'
    for i in range(1, len(names)):
      where_clause += f' OR {column_name} LIKE @{column_name}{i}'
    where_clause += ')'
    return where_clause
  # TODO: Maybe make the cutoff date configurable.
  cutoff_date = (datetime.datetime.now() - datetime.timedelta(
      days=30)).strftime('%Y-%m-%d')
  query = QUERY.format(**{
    'job_table_name': os.environ['JOB_HISTORY_TABLE_NAME'],
    'metric_table_name': os.environ['METRIC_HISTORY_TABLE_NAME'],
    'test_name_where_clause': _make_where_clause('test_name', test_names),
    'metric_name_where_clause': _make_where_clause(
        'metric_name', metric_names),
    'cutoff_date': cutoff_date,
  })
  return query

def fetch_data(test_names, metric_names):
  if not test_names or not metric_names:
    raise ValueError('Neither test_names nor metric_names can be empty.')
  dataframe = utils.run_query(
    get_query(test_names, metric_names),
    cache_key=('metrics-{}-{}'.format(str(test_names), str(metric_names))),
    config=get_query_config(test_names, metric_names))
  return dataframe

def make_html_table(data_grid):
  if not data_grid:
    return ''
  cell_width = 100
  normal_style = f'style="width:{cell_width}px; border:1px solid #cfcfcf"'
  alert_style = f'style="width:{cell_width}px; border:1px solid #cfcfcf; background-color: #ff8a8a"'
  table_width = cell_width * len(data_grid[0])
  table_html = f'<table style="width:{table_width}px">'
  first_row = True
  for row in data_grid:
    values = []
    for col in row:
      try:
        values.append(float(col))
      except ValueError:
        continue
    # First row uses '<th>' HTML tag, others use '<td>'.
    tag_type = 'h' if first_row else 'd'
    table_html += '<tr>'
    for col in row:
      # Use normal cell style unless the cell's value is unusually high or low.
      style = normal_style
      try:
        v = float(col)
        # Find the mean/stddev of this row's values but make sure to exclude
        # the current value.
        values_copy = list(values)
        for i, x in enumerate(values_copy):
          if math.isclose(x, v):
            values_copy.pop(i)
        mean = np.mean(values_copy) if len(values_copy) >= 5 else 0
        stddev = np.std(values_copy) if len(values_copy) >= 5 else 0
        if stddev > 0 and (abs(v - mean) / stddev) > 5.0:
          style = alert_style
      except ValueError:
        pass
      table_html += f'<t{tag_type} {style}>{col}</t{tag_type}>'
    table_html += '</tr>'
    first_row = False
  table_html += '</table>'
  return table_html

def make_plots(test_names, metric_names, dataframe):
  if not dataframe['metric_name'].any():
    logging.error("Found no data: {}\n{}".format(test_names, dataframe))
    return

  # Split data into 1 dataframe per metric so we can easily make 1 graph
  # per metric.
  metric_to_dataframe = {}
  for metric_name in set(np.unique(dataframe['metric_name']).tolist()):
    metric_dataframe = dataframe[dataframe['metric_name'] == metric_name]
    metric_to_dataframe[metric_name] = metric_dataframe

  all_rows = []
  for metric_name, dataframe in metric_to_dataframe.items():
    plot, table_row = _make_plot_and_table(
        metric_name, metric_to_dataframe[metric_name])
    all_rows.extend([plot, table_row])
  return all_rows


def _make_plot_and_table(metric_name, dataframe):
  # Record some global stats about the entire suite of tests.
  all_dates = np.unique(dataframe['run_date']).tolist()[-1::-1]
  y_max = 1.1 * dataframe['metric_value'].max()
  y_min = 0.9 * dataframe['metric_value'].min()
  if y_max == 0 and y_min == 0:
    y_max = 1.0
    y_min = 0.0

  # Split each test into its own dataframe.
  test_name_to_df = {}
  for test_name in np.unique(dataframe['test_name']).tolist():
    test_name_to_df[test_name] = dataframe[dataframe['test_name'] == test_name]

  tooltip_template = """
    Value: @test_name<br/>Metric Value: @metric_value<br/>Job status: @job_status<br/>Date: @run_date"""
  plot = figure(
      title=metric_name,
      plot_width=100*len(all_dates),
      y_range=(y_min, y_max),
      x_range=all_dates,
      toolbar_location=None,
      tools="tap",
      tooltips=tooltip_template)

  color_mapper = CategoricalColorMapper(
      factors=['success', 'failure', 'timeout'],
      palette=['#000000', '#ffffff', '#ffffff'])

  all_tables = []
  for test, df in test_name_to_df.items():
    source = ColumnDataSource(data=df)
    line = plot.line(
        x='run_date', y='metric_value', line_width=3, color='#000000',
        source=source)
    plot.circle(
        x='run_date',
        y='metric_value',
        source=source,
        fill_color={'field': 'job_status', 'transform': color_mapper},
        size=15)

  # Create a table representation of the data from the plot above.
  # Each date is a column and each row is a test.
  # Add an extra first row for the headers and an extra first
  # column for the test name.
  data_grid = [['-' for _ in range(len(all_dates) + 1)] for _ in range(len(
      test_name_to_df.keys()) + 1)]
  data_grid[0] = ['Test'] + all_dates
  # Offset by 1 to account for the test name column at index 0.
  run_date_to_column_index = {date: index + 1 for index, date in enumerate(
      all_dates)}
  for row_i, test_name in enumerate(test_name_to_df.keys()):
    row_i += 1  # Offset by 1 to account for the header row at index 0.
    data_grid[row_i][0] = \
        f"""<a href="metrics?test_name={test_name}">{test_name}</a>"""
    for row in test_name_to_df[test_name].iterrows():
      metric_value = row[1]['metric_value']
      run_date = row[1]['run_date']
      data_grid[row_i][run_date_to_column_index[run_date]] = \
          f'{metric_value:0.2f}'
  table = make_html_table(data_grid)
  table_row = Div(text=table)

  plot.xaxis.major_label_orientation = math.pi / 3
  return plot, table_row

