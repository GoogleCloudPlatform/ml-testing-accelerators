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
all_test_names = run_query(
    QUERY, cache_key=('xlmltest'))['test_name'].values.tolist()
test_names = []
for name in all_test_names:
  for prefix in test_name_prefixes:
    if prefix in name:
      test_names.append(name)


def update(attr, old, new):
  if old == new:
    return
  t0 = time.time()
  timer = Paragraph()
  timer.text = '(Executing query...)'
  test_name_para = Paragraph()
  test_name_para.text = 'Metrics for: {}'.format(test_select.value)
  curdoc().clear()
  base_rows = [row(test_name_para), row(test_select, metric_select, timer)]
  curdoc().add_root(
      column(
          children=base_rows,
      )
  )
  test_name = test_select.value
  if test_name not in test_names:
    timer.text = 'Invalid test_name: {}'.format(test_name)
    return
  metric_substr = metric_select.value
  cutoff_timestamp = (datetime.datetime.now() - datetime.timedelta(
      days=30)).strftime('%Y-%m-%d %H:%M:%S UTC')
  data = metric_history.fetch_data(test_name, cutoff_timestamp)
  plots = metric_history.make_plots(test_name, metric_substr, data)
  plot_rows = [row(p) for p in plots] if plots else []
  curdoc().clear()
  curdoc().add_root(
      column(
          children=base_rows + plot_rows,
      )
  )
  t1 = time.time()
  timer.text = '(Execution time: %s seconds)' % round(t1 - t0, 4)

# Try to parse the requested test_name from URL args.
args = curdoc().session_context.request.arguments
current_test_name = test_names[1]
if args and 'test_name' in args:
  passed_in_test_name = str(args['test_name'][0], 'utf-8')
  current_test_name = passed_in_test_name

test_select = Select(title='Select a test:', value=current_test_name, options=test_names)
test_select.on_change('value', update)
metric_select = TextInput(value="", title="Metric substring (blank to see all). Press enter.")
metric_select.on_change('value', update)

curdoc().title = 'Metrics History'
update('value', '', current_test_name)

