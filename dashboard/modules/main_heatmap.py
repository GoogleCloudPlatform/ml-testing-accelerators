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

from bokeh.core.properties import Instance
from bokeh.events import DoubleTap
from bokeh.models import ColumnDataSource, CategoricalColorMapper, CustomJS, TapTool
from bokeh.plotting import figure

import javascript_utils
import utils

import numpy as np
import pandas as pd

from absl import logging


JOB_STATUS_QUERY = """
SELECT
  x.test_name,
  x.job_status,
  SAFE_CAST(EXTRACT(date from x.timestamp) AS STRING) AS run_date,
  x.stackdriver_logs_link AS logs_link,
  x.uuid
FROM (
  SELECT
    test_name,
    SAFE_CAST(EXTRACT(date from timestamp) AS STRING) as run_date,
    max(farm_fingerprint(uuid)) as max_uuid,
  FROM
    `xl-ml-test.metrics_handler_dataset.job_history`
  WHERE
    test_name like '%(test_name_prefix)s%%'
  GROUP BY
    test_name, run_date
) AS y 
INNER JOIN
  `xl-ml-test.metrics_handler_dataset.job_history` AS x
ON
  y.test_name = x.test_name AND
  y.max_uuid = farm_fingerprint(x.uuid)
ORDER BY
  run_date DESC
"""


METRIC_STATUS_QUERY = """
SELECT
  test_name,
  SAFE_CAST(EXTRACT(date from timestamp) AS STRING) as run_date,
  metric_name,
  metric_value,
  metric_upper_bound,
  metric_lower_bound
FROM
  `xl-ml-test.metrics_handler_dataset.metric_history`
WHERE
  test_name LIKE '%(test_name_prefix)s%%' AND
  (metric_lower_bound IS NOT NULL OR metric_upper_bound IS NOT NULL) AND
  (metric_value < metric_lower_bound OR metric_value > metric_upper_bound)
LIMIT 1000
"""

COLORS = {'success': '#02cf17', 'failure': '#a10606'}

def fetch_data(test_name_prefix):
  dataframe = utils.run_query(
    JOB_STATUS_QUERY % {'test_name_prefix': test_name_prefix},
    cache_key=('job-status-%s' % test_name_prefix))
  metrics_dataframe = utils.run_query(
    METRIC_STATUS_QUERY % {'test_name_prefix': test_name_prefix},
    cache_key=('metric-status-%s' % test_name_prefix))

  # Collect all test+date combinations where metrics were out of bounds.
  oob_tests = collections.defaultdict(list)
  def _test_date_key(test, date):
    return '{}:{}'.format(test, date)
  for row in metrics_dataframe.iterrows():
    oob_test_name = row[1]['test_name']
    oob_run_date = row[1]['run_date']
    oob_metric_name = row[1]['metric_name']
    oob_upper_bound = row[1]['metric_upper_bound']
    oob_lower_bound = row[1]['metric_lower_bound']
    failure_explanation = (
        f'Metric `{oob_metric_name}` was outside expected bounds of: '
        f'({oob_lower_bound}, {oob_upper_bound})')
    oob_tests[_test_date_key(oob_test_name, oob_run_date)].append(
        failure_explanation)

  dataframe['overall_status'] = dataframe['job_status'].apply(
      lambda x: x)

  # Record the status of the metrics for every test.
  dataframe['failed_metrics'] = dataframe['job_status'].apply(
    lambda x: [])
  for row in dataframe.iterrows():
    test_name = row[1]['test_name']
    failed_metrics = oob_tests.get(_test_date_key(
        test_name, row[1]['run_date'])) or []
    if failed_metrics:
      dataframe['failed_metrics'][row[0]] = failed_metrics
      dataframe['overall_status'][row[0]] = 'failure'
            
  # Create a few convenience columns to use in the dashboard.
  dataframe['job_status_abbrev'] = dataframe['overall_status'].apply(
      lambda x: '' if x.startswith('success') else x[:1].upper())
  dataframe['metrics_link'] = dataframe['test_name'].apply(
      lambda x: 'metrics?test_name={}'.format(x))
  dataframe['logs_download_command'] = dataframe['logs_link'].apply(
      utils.get_download_command)

  return dataframe


def make_plot(dataframe):
  logging.error('Len of dataframe: {}'.format(len(dataframe)))
  source = ColumnDataSource(data=dataframe)
  all_dates = np.unique(source.data['run_date']).tolist()
  MIN_DATES_TO_RENDER = 15
  if len(all_dates) < MIN_DATES_TO_RENDER:
    all_dates.extend(['0-spacer{:02d}'.format(x) for x in range(
        MIN_DATES_TO_RENDER - len(all_dates))])

  # Remove duplicate dates.
  all_dates = sorted(list(set(all_dates)))

  all_test_names = np.unique(source.data['test_name']).tolist()
  longest_test_name = max(len(name) for name in all_test_names)

  tooltip_template = """@overall_status on @run_date"""
  plot = figure(
      plot_width=(2*longest_test_name)+(45*len(all_dates)),
      plot_height=30*len(all_test_names),
      x_range = all_dates[-1::-1],  # Reverse for latest dates first.
      x_axis_location='above',
      y_range = all_test_names[-1::-1],  # Reverse for alphabetical.
      tools="tap",
      toolbar_location=None,
      tooltips=tooltip_template)

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
      x="run_date", y="test_name", width=1, height=1,
      source=source, line_color='#ffffff', line_width=1.5,
      fill_color={'field': 'overall_status', 'transform': color_mapper})

  onclick_code = javascript_utils.get_modal_javascript('pass_fail_grid')
  taptool = plot.select(type=TapTool)
  taptool.callback = CustomJS(
      args=dict(source=source),
      code=onclick_code)

  # Add the 1-letter 'F' codes for failed jobs onto the grid of rectangles.
  plot.text(
      x='run_date',
      y='test_name',
      text='job_status_abbrev',
      source=source,
      text_color='#ffffff',
      x_offset=-5,
      y_offset=10,
      text_font_style='bold')
  return plot
