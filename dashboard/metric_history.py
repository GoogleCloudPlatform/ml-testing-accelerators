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

from math import pi

from bokeh.models import CategoricalColorMapper, ColumnDataSource, CustomJS, HoverTool, TapTool, Whisker
from bokeh.plotting import figure

import javascript_utils
import utils

import numpy as np


# For a given test_name, find the history of metrics for that test.
QUERY = """
SELECT
  metrics.test_name,
  metrics.metric_name,
  SAFE_CAST(DATE(metrics.timestamp, 'US/Pacific') AS STRING) AS run_date,
  metrics.metric_value,
  metrics.metric_lower_bound,
  metrics.metric_upper_bound,
  job.job_status,
  job.stackdriver_logs_link AS logs_link
FROM (
  SELECT
    x.test_name,
    x.metric_name,
    x.timestamp,
    x.metric_value,
    x.metric_lower_bound,
    x.metric_upper_bound,
    x.uuid
  FROM (
    SELECT
      test_name,
      metric_name,
      SAFE_CAST(DATE(timestamp, 'US/Pacific') AS STRING) AS run_date,
      min(timestamp) as min_timestamp
    FROM
      `xl-ml-test.metrics_handler_dataset.metric_history`
    WHERE
      test_name = @test_name AND
      (metric_name NOT LIKE '%%__Percentile_%%' OR metric_name LIKE '%%__Percentile_99%%')
    GROUP BY
      test_name, metric_name, run_date
  ) AS y
  INNER JOIN `xl-ml-test.metrics_handler_dataset.metric_history` AS x
  ON
    y.test_name = x.test_name AND
    y.metric_name = x.metric_name AND
    y.min_timestamp = x.timestamp
) AS metrics
INNER JOIN `xl-ml-test.metrics_handler_dataset.job_history` AS job
ON
  metrics.uuid = job.uuid
ORDER BY
  run_date DESC
"""

def _get_query_config(test_name):
  return {
    'query': {
      'parameterMode': 'NAMED',
      'queryParameters': [
        {
          'name': 'test_name',
          'parameterType': {'type': 'STRING'},
          'parameterValue': {'value': test_name},
        },
      ]
    }
  }

def fetch_data(test_name):
  dataframe = utils.run_query(
    QUERY,
    cache_key=('metrics-%s' % test_name),
    config=_get_query_config(test_name))

  dataframe['logs_download_command'] = dataframe['logs_link'].apply(
      utils.get_download_command)
  return dataframe


def make_plots(test_name, metric_name_substr, dataframe):
  if not dataframe['metric_name'].any():
    print("FOUND NO DATA: {}\n{}".format(test_name, dataframe))
    return

  # Split data into 1 dataframe per metric so we can easily make 1 graph
  # per metric.
  metric_to_dataframe = {}
  all_metric_names = set()
  oob_metric_names = set()
  metric_name_substr = metric_name_substr.lower()
  for metric_name in set(np.unique(dataframe['metric_name']).tolist()):
    if metric_name_substr not in metric_name.lower():
      continue
    all_metric_names.add(metric_name)
    metric_dataframe = dataframe[dataframe['metric_name'] == metric_name]
    metric_to_dataframe[metric_name] = metric_dataframe

    # Determine if this metric was out of bounds on its most recent test run.
    all_dates = sorted(np.unique(metric_dataframe['run_date']).tolist())
    for row in metric_dataframe.iterrows():
      run_date = row[1]['run_date']
      if run_date == all_dates[-1]:
        metric_name = row[1]['metric_name']
        metric_value = row[1]['metric_value']
        upper_bound = row[1]['metric_upper_bound']
        lower_bound = row[1]['metric_lower_bound']
        if (not np.isnan(upper_bound) and metric_value > upper_bound) or (
            not np.isnan(lower_bound) and metric_value < lower_bound):
          oob_metric_names.add(metric_name) 

  # Render the out-of-bounds metrics first so they're easier to find.
  all_plots = []
  for metric_name in oob_metric_names:
    all_plots.append(
        _make_plot(metric_name, metric_to_dataframe[metric_name], oob=True))
  for metric_name in all_metric_names - oob_metric_names:
    all_plots.append(
        _make_plot(metric_name, metric_to_dataframe[metric_name], oob=False))
  return all_plots


def _make_plot(metric_name, dataframe, oob=False):
  source = ColumnDataSource(data=dataframe)

  # For some metrics, we'll want to show upper+lower bounds as error bars
  # or the bound as a line if only upper bound or only lower bound exists.
  # If the metric has no bounds at all, then we'll skip these visual aids.
  found_upper_bound = False
  found_lower_bound = False
  bound_value_and_date = []
  for bound_name in ['metric_upper_bound', 'metric_lower_bound']:
    for i, x in enumerate(source.data[bound_name]):
      if np.isnan(x):
        source.data[bound_name].itemset(i, source.data['metric_value'].item(i))
      else:
        bound_value_and_date.append((source.data['run_date'].item(i), x))
        if bound_name == 'metric_upper_bound':
          found_upper_bound = True
        elif bound_name == 'metric_lower_bound':
          found_lower_bound = True

  # We want to add whiskers or bound line but not both.
  should_add_whiskers = False
  should_add_bound_line = False
  if found_upper_bound and found_lower_bound:
    should_add_whiskers = True
  elif found_upper_bound or found_lower_bound:
    should_add_bound_line = True
    bound_line_y_value = sorted(
        bound_value_and_date, key=lambda x: x[0], reverse=True)[0][1]

  y_max = 1.1 * max(source.data['metric_upper_bound'].max(),
                    source.data['metric_value'].max())
  y_min = 0.9 * min(source.data['metric_lower_bound'].min(),
                    source.data['metric_value'].min())
  if y_max == 0 and y_min == 0:
    y_max = 1.0
    y_min = 0.0
  all_dates = np.unique(source.data['run_date']).tolist()

  tooltip_template = """
    Value: @metric_value<br/>Job status: @job_status<br/>Date: @run_date"""
  plot = figure(
      title=metric_name,
      plot_width=100*len(all_dates),
      y_range=(y_min, y_max),
      x_range=all_dates[-1::-1],
      toolbar_location=None,
      tools="tap",
      tooltips=tooltip_template)
  line = plot.line(
      x='run_date', y='metric_value', line_width=3, color='#000000',
      source=source)

  color_mapper = CategoricalColorMapper(
      factors=['success', 'failure', 'timeout'],
      palette=['#000000', '#ffffff', '#ffffff'])

  plot.circle(
      x='run_date',
      y='metric_value',
      source=source,
      fill_color={'field': 'job_status', 'transform': color_mapper},
      size=15)

  onclick_code = javascript_utils.get_modal_javascript(
      'metrics_history')
  taptool = plot.select(type=TapTool)
  taptool.callback = CustomJS(
      args=dict(source=source),
      code=onclick_code)

  if should_add_whiskers:
    plot.add_layout(Whisker(
        source=source, base='run_date', upper='metric_upper_bound',
        lower='metric_lower_bound', level='overlay'))
  elif should_add_bound_line:
    plot.line(source=source, x='run_date', y=bound_line_y_value,
              line_dash='dashed', line_width=2)
  plot.xaxis.major_label_orientation = pi / 3
  if oob:
    plot.outline_line_width = 7
    plot.outline_line_color = "red"
  return plot

