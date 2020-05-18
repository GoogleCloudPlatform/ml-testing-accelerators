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

import metric_compare

from utils import run_query

def update(attr, old, new):
  t0 = time.time()
  timer = Paragraph()
  timer.text = '(Executing query...)'
  curdoc().clear()
  base_rows = [row(timer), row(test_select), row(metric_select)]
  curdoc().add_root(
      column(
          children=base_rows,
      )
  )
  test_names = [x for x in test_select.value.split(',') if x]
  print('TEST_NAMES: __{}__'.format(test_names))
  metric_names = [x for x in metric_select.value.split(',') if x]
  print('METRIC_NAMES: __{}__'.format(metric_names))
  if not test_names or not metric_names:
    timer.text = 'Neither test_names nor metric_names can be blank.'
    return
  timer.text = 'TEST_NAMES: {}      METRIC_NAMES: {}'.format(test_names, metric_names)
  data = metric_compare.fetch_data(test_names, metric_names)
  plots = metric_compare.make_plots(test_names, metric_names, data)
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
import base64
current_test_names = base64.b64decode(args.get('test_names', [b''])[0]).decode('utf-8')
current_metric_names = base64.b64decode(args.get('metric_names', [b''])[0]).decode('utf-8')
print('done parsing args')

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

