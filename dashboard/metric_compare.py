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
from math import pi
import os

from bokeh.layouts import column, row
from bokeh.models import CategoricalColorMapper, ColumnDataSource, DataTable, Div, HoverTool, NumberFormatter, OpenURL, PreText, TableColumn, TapTool, Whisker
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
  def _make_where_clause(column_name, names):
    where_clause = f'({column_name} LIKE @{column_name}0'
    for i in range(1, len(names)):
      where_clause += f' OR {column_name} LIKE @{column_name}{i}'
    where_clause += ')'
    return where_clause
  # TODO: Maybe make the cutoff date configurable.
  cutoff_date = (datetime.datetime.now() - datetime.timedelta(30)).strftime(
      '%Y-%m-%d')
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

def make_table(data_grid):
  if not data_grid:
    return ''
  STYLE = 'style="width:100px; border:1px solid #cfcfcf"'
  table_width = 100 * len(data_grid[0])
  table_html = f'<table style="width:{table_width}px">'
  first_row = True
  for row in data_grid:
    # First row uses '<th>' HTML tag, others use '<td>'.
    tag_type = 'h' if first_row else 'd'
    table_html += '<tr>'
    for col in row:
      table_html += f'<t{tag_type} {STYLE}>{col}</t{tag_type}>'
    table_html += '</tr>'
    first_row = False
  table_html += '</table>'
  return table_html

def make_plots(test_names, metric_names, dataframe):
  if not dataframe['metric_name'].any():
    print("FOUND NO DATA: {}\n{}".format(test_names, dataframe))
    return

  # Split data into 1 dataframe per metric so we can easily make 1 graph
  # per metric.
  metric_to_dataframe = {}
  for metric_name in set(np.unique(dataframe['metric_name']).tolist()):
    metric_dataframe = dataframe[dataframe['metric_name'] == metric_name]
    metric_to_dataframe[metric_name] = metric_dataframe

  all_plots = []
  for metric_name, dataframe in metric_to_dataframe.items():
    plot, tables_row = _make_plot_and_tables(
        metric_name, metric_to_dataframe[metric_name])
    all_plots.extend([plot, tables_row])
  return all_plots


def _make_plot_and_tables(metric_name, dataframe):
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
    #table = DataTable(selectable=True, source=source, width=275, height=275, columns=[
    #          TableColumn(field="metric_value", title=metric_name, width=80, formatter=NumberFormatter(format="0,0")),
    #          TableColumn(field="run_date", title="Date", width=80),
    #])
    #all_tables.append(column(
    #    Div(text=f"""<a href="metrics?test_name={test}">{test}</a>""",
    #        width=275, height=10),
    #    table))
  #tables_rows = []
  #chunked_tables = np.array_split(np.array(all_tables), len(all_tables)//5)
  #print(len(chunked_tables))
  #print(len(chunked_tables[0]))
  #for chunk in chunked_tables:
  #  tables_rows.append(row(column(chunk.tolist())))
  #tables_row = row(tables_rows)

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
  table = make_table(data_grid)
  print(table)
  table_row = Div(text=table)



  #taptool = plot.select(type=TapTool)
  #test_name='mnist'
  #taptool.callback = OpenURL(url='metrics?test_name='+test_name)

  #tables_row = PreText(text=str(dataframe), width=1000)

  plot.xaxis.major_label_orientation = pi / 3
  return plot, table_row

