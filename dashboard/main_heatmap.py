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

import collections
import math
import os
import urllib.parse

from bokeh.core.properties import Instance
from bokeh.events import DoubleTap
from bokeh.models import ColumnDataSource, CategoricalColorMapper, CustomJS, TapTool, HoverTool
from bokeh.plotting import figure

import javascript_utils
import utils

import numpy as np
import pandas as pd

from absl import logging

JOB_HISTORY_TABLE_NAME = os.environ['JOB_HISTORY_TABLE_NAME']
METRIC_HISTORY_TABLE_NAME = os.environ['METRIC_HISTORY_TABLE_NAME']

JOB_STATUS_QUERY = f"""
SELECT 
  test_name,
  job_status,
  SAFE_CAST(DATE(timestamp, 'US/Pacific') AS STRING) as run_date,
  logs_link,
  workload_link,
  uuid
FROM (
  SELECT
    test_name,
    job_status,
    timestamp,
    FIRST_VALUE (timestamp) OVER (
        PARTITION BY
            test_name, SAFE_CAST(DATE(timestamp, 'US/Pacific') AS STRING)
        ORDER BY 
            timestamp
    ) AS first_time,
    logs_link,
    workload_link,
    uuid
  FROM (
    SELECT 
      uuid,
      test_name,
      job_status,
      timestamp,
      stackdriver_logs_link AS logs_link,
      kubernetes_workload_link AS workload_link
    FROM `{JOB_HISTORY_TABLE_NAME}`
    WHERE
      test_name like @test_name_prefix AND
      timestamp >= @cutoff_timestamp
  )
)
WHERE timestamp = first_time
"""

METRIC_STATUS_QUERY = f"""
SELECT
  test_name,
  SAFE_CAST(DATE(timestamp, 'US/Pacific') AS STRING) AS run_date,
  metric_name,
  metric_value,
  metric_upper_bound,
  metric_lower_bound
FROM
  `{METRIC_HISTORY_TABLE_NAME}`
WHERE
  test_name LIKE @test_name_prefix AND
  (metric_lower_bound IS NOT NULL OR metric_upper_bound IS NOT NULL) AND
  (metric_value < metric_lower_bound OR metric_value > metric_upper_bound) AND
  timestamp >= @cutoff_timestamp
LIMIT 1000
"""

def _get_query_config(test_name_prefix, cutoff_timestamp):
  return {
    'query': {
      'parameterMode': 'NAMED',
      'queryParameters': [
        {
          'name': 'test_name_prefix',
          'parameterType': {'type': 'STRING'},
          'parameterValue': {'value': '{}%'.format(test_name_prefix)},
        },
        {
          'name': 'cutoff_timestamp',
          'parameterType': {'type': 'TIMESTAMP'},
          'parameterValue': {'value': cutoff_timestamp},
        },
      ]
    }
  }

COLORS = {'success': '#02cf17', 'failure': '#a10606'}

def fetch_data(test_name_prefix, cutoff_timestamp):
  job_status_dataframe = utils.run_query(
    JOB_STATUS_QUERY,
    cache_key=('job-status-%s' % test_name_prefix),
    config=_get_query_config(test_name_prefix, cutoff_timestamp))
  metrics_dataframe = utils.run_query(
    METRIC_STATUS_QUERY,
    cache_key=('metric-status-%s' % test_name_prefix),
    config=_get_query_config(test_name_prefix, cutoff_timestamp))
  combined_dataframe = process_dataframes(
      job_status_dataframe, metrics_dataframe)
  return combined_dataframe

def process_dataframes(job_status_dataframe, metrics_dataframe):
  if job_status_dataframe.empty or 'job_status' not in job_status_dataframe:
    return pd.DataFrame()

  # Default to logs tab of Kubernetes workload
  def _append_workload_logs_path(url):
    parsed = urllib.parse.urlparse(url)
    parsed = parsed._replace(path=parsed.path + '/logs')
    return urllib.parse.urlunparse(parsed)

  job_status_dataframe['workload_link'] = (
    job_status_dataframe['workload_link'].map(_append_workload_logs_path))

  # Collect all test+date combinations where metrics were out of bounds.
  oob_tests = collections.defaultdict(list)
  def _test_date_key(test, date):
    return '{}:{}'.format(test, date)
  for row in metrics_dataframe.iterrows():
    oob_test_name = row[1]['test_name']
    oob_run_date = row[1]['run_date']
    oob_metric_name = row[1]['metric_name']
    oob_metric_value = row[1]['metric_value']
    oob_upper_bound = row[1]['metric_upper_bound']
    oob_lower_bound = row[1]['metric_lower_bound']
    failure_explanation = (
        f'Metric `{oob_metric_name}` was outside expected bounds of: '
        f'({oob_lower_bound}, {oob_upper_bound}) with value of '
        f'{oob_metric_value:.2f}')
    oob_tests[_test_date_key(oob_test_name, oob_run_date)].append(
        failure_explanation)

  job_status_dataframe['overall_status'] = job_status_dataframe[
      'job_status'].apply(lambda x: x)
  job_status_dataframe['detailed_status'] = job_status_dataframe[
      'job_status'].apply(lambda x: '{}_in_job'.format(x))

  # Record the status of the metrics for every test.
  job_status_dataframe['failed_metrics'] = job_status_dataframe[
      'job_status'].apply(lambda x: [])
  for row in job_status_dataframe.iterrows():
    test_name = row[1]['test_name']
    job_status = row[1]['job_status']
    failed_metrics = oob_tests.get(_test_date_key(
        test_name, row[1]['run_date'])) or []
    if failed_metrics and job_status == 'success':
      job_status_dataframe['failed_metrics'][row[0]] = failed_metrics
      job_status_dataframe['overall_status'][row[0]] = 'failure'
      job_status_dataframe['detailed_status'][row[0]] = 'failure_in_metrics'

  # Create a few convenience columns to use in the dashboard.
  def _get_single_char_test_status(detailed_status):
    if detailed_status.startswith('success'):
      return ''
    elif detailed_status.startswith('failure') and 'metrics' in \
        detailed_status:
      return 'M'
    elif detailed_status.startswith('missed'):
      # `X` is used to denote missed executions because `M` is already in use by
      # metrics failures.
      return 'X'
    else:
      return detailed_status[:1].upper()
  job_status_dataframe['job_status_abbrev'] = job_status_dataframe[
      'detailed_status'].apply(_get_single_char_test_status)
  job_status_dataframe['metrics_link'] = job_status_dataframe[
      'test_name'].apply(lambda x: 'metrics?test_name={}'.format(x))

  return job_status_dataframe

def make_plot(dataframe):
  if 'run_date' not in dataframe:
    return None  # No dates to render.
  dataframe['display_date'] = dataframe.run_date
  dataframe.run_date = pd.to_datetime(dataframe.run_date)
  latest_results = dataframe.loc[dataframe.reset_index().groupby(['test_name'])['run_date'].idxmax()]
  latest_results.display_date = 'latest'
  dataframe = dataframe.append(latest_results)
  source = ColumnDataSource(data=dataframe)
  all_dates = np.unique(source.data['display_date']).tolist()
  if not all_dates:
    return None  # No dates to render.

  # The heatmap doesn't render correctly if there are very few dates.
  MIN_DATES_TO_RENDER = 15
  if len(all_dates) < MIN_DATES_TO_RENDER:
    all_dates.extend(['0-spacer{:02d}'.format(x) for x in range(
        MIN_DATES_TO_RENDER - len(all_dates))])

  # Remove duplicate dates.
  all_dates = sorted(list(set(all_dates)))

  all_test_names = np.unique(source.data['test_name']).tolist()
  longest_test_name = max(len(name) for name in all_test_names)

  plot = figure(
      plot_width=(2*longest_test_name)+(45*len(all_dates)),
      plot_height=100 + 30*len(all_test_names),
      x_range = all_dates[-1::-1],  # Reverse for latest dates first.
      x_axis_location='above',
      y_range = all_test_names[-1::-1],  # Reverse for alphabetical.
      tools="tap",
      toolbar_location=None)
  plot.add_tools(
    HoverTool(
      tooltips=[
        ('status', '@overall_status'),
        ('date', '@run_date{%F}'),
      ],
      formatters={
        '@run_date': 'datetime',
      }
    )
  )

  plot.grid.grid_line_color = None
  plot.axis.axis_line_color = None
  plot.axis.axis_label_text_align = 'left'
  plot.axis.major_tick_line_color = None
  plot.axis.major_label_text_font_size = "10pt"
  plot.axis.major_label_standoff = 0
  plot.xaxis.major_label_orientation = -(math.pi / 3)

  color_mapper = CategoricalColorMapper(
      factors=['success', 'failure', 'timeout'],
      palette=[COLORS['success'], COLORS['failure'], COLORS['failure']])
  rect = plot.rect(
      x="display_date", y="test_name", width=1, height=1,
      source=source, line_color='#ffffff', line_width=1.5,
      fill_color={'field': 'overall_status', 'transform': color_mapper})

  onclick_code = javascript_utils.get_modal_javascript('pass_fail_grid')
  taptool = plot.select(type=TapTool)
  taptool.callback = CustomJS(
      args=dict(source=source),
      code=onclick_code)

  # Add the 1-letter 'F' codes for failed jobs onto the grid of rectangles.
  plot.text(
      x='display_date',
      y='test_name',
      text='job_status_abbrev',
      source=source,
      text_color='#ffffff',
      x_offset=-5,
      y_offset=10,
      text_font_style='bold')
  return plot
