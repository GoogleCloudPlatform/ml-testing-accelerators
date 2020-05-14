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

import logging
import os
import time

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Select, Paragraph, Panel, Tabs, TextInput

import metric_history as metric_history

from utils import run_query
QUERY = """
    SELECT DISTINCT(test_name)
    FROM
      `xl-ml-test.metrics_handler_dataset.metric_history`
    ORDER BY TEST_NAME
"""
test_name_prefixes = os.environ.get('TEST_NAME_PREFIXES', '').split(',')
all_valid_test_names = run_query(
    QUERY, cache_key=('xlmltest'))['test_name'].values.tolist()
valid_test_names = []
for name in all_valid_test_names:
  for prefix in test_name_prefixes:
    if prefix in name:
      valid_test_names.append(name)


def update(attr, old, new):
  if old == new:
    return
  t0 = time.time()
  timer = Paragraph()
  timer.text = '(Executing query...)'
  curdoc().clear()
  base_rows = [row(test_select, metric_select, timer)]
  curdoc().add_root(
      column(
          children=base_rows,
      )
  )
  test_names = test_select.value.split(',')
  for name in test_names:
    if name not in valid_test_names:
      timer.text = 'Invalid test_name: {}'.format(test_name)
      return
  metric_names = metric_select.value.split(',')
  if not test_names or not metric_names:
    timer.text = 'Neither test_names nor metric_names can be blank.'
    return
  timer.text = 'TEST_NAMES: {}      METRIC_NAMES: {}'.format(test_names, metric_names)
  return
  data = metric_compare.fetch_data(test_names, metric_names)
  plots = metric_history.make_plots(test_names, metric_names, data)
  plot_rows = [row(p) for p in plots] if plots else []
  curdoc().clear()
  curdoc().add_root(
      column(
          children=base_rows + plot_rows,
      )
  )
  t1 = time.time()
  timer.text = '(Execution time: %s seconds)' % round(t1 - t0, 4)

# Try to parse test_names and metric_names from URL args.
args = curdoc().session_context.request.arguments or {}
current_test_names = str(args.get('test_names', [''])[0], 'utf-8')
current_metric_names = str(args.get('metric_names', [''])[0], 'utf-8')

test_select = TextInput(
    value=current_test_names,
    title="Comma-separated test names. Press enter to redraw.")
test_select.on_change('value', update)
metric_select = TextInput(
    value=current_metric_names,
    title="Comma-separated metric names. Press enter to redraw.")
metric_select.on_change('value', update)

curdoc().title = 'Compare Metrics'
update('value', '', current_test_names)

