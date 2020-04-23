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

# Run with:
#   python3 -m bokeh serve --show dashboard.py metrics.py

import logging
import concurrent
import time

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Paragraph, Panel, Tabs

import main_heatmap


def parallel_fetch_data(test_name_prefixes):
  """Fetch data from BigQuery for each dashboard tab in parallel.

  Args:
    test_name_prefixes(list[string]): 1 string for each desired tab. Each string is
      used to query for tests that begin with that prefix.

  Returns:
    data(dict): Key is the test_name_prefix and value is the result of the query
      from BigQuery.
  """
  t0 = time.time()
  # Create a thread pool: one separate thread for each query to run.
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(test_name_prefixes)) as executor:
    # Prepare the thread tasks.
    tasks = {}
    for test_name_prefix in test_name_prefixes:
      task = executor.submit(main_heatmap.fetch_data, test_name_prefix)
      tasks[task] = test_name_prefix

    # Run the tasks and collect results as they arrive.
    data = {}
    for task in concurrent.futures.as_completed(tasks):
      key = tasks[task]
      data[key] = task.result()
  # Return results once all tasks have been completed.
  t1 = time.time()
  timer.text = '(Execution time: %s seconds)' % round(t1 - t0, 4)
  return data


timer = Paragraph()

all_tabs = []
# TODO: Pass tabs/test prefix via config.
test_name_prefixes = ['pt-nightly', 'pt-1.5', 'tf-nightly']
all_data = parallel_fetch_data(test_name_prefixes)
for test_prefix, data in all_data.items():
  plot = main_heatmap.make_plot(data)
  all_tabs.append(Panel(child=plot, title=test_prefix))

curdoc().title = "Test pass/fail Dashboard"
curdoc().add_root(column(children=[row(timer), row(Tabs(tabs=all_tabs))]))
