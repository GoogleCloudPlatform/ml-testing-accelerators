# Lint as: python3
"""Tests for main_heatmap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import datetime
import metric_compare
import numpy as np
import pandas as pd
import os


JOB_TABLE_NAME = 'jh'
METRIC_TABLE_NAME = 'mh'

class MetricHistoryTest(parameterized.TestCase):
  def setUp(self):
    os.environ['JOB_HISTORY_TABLE_NAME'] = JOB_TABLE_NAME
    os.environ['METRIC_HISTORY_TABLE_NAME'] = METRIC_TABLE_NAME

  def tearDown(self):
    os.environ.pop('JOB_HISTORY_TABLE_NAME')
    os.environ.pop('METRIC_HISTORY_TABLE_NAME')

  def test_get_query_and_config_one_element(self):
    test_names = ['tf-nightly-mnist']
    metric_names = ['acc', 'loss']
    query = metric_compare.get_query(test_names, metric_names)
    config = metric_compare.get_query_config(test_names, metric_names)
    self.assertEqual(query, metric_compare.QUERY.format(**{
      'job_table_name': JOB_TABLE_NAME,
      'metric_table_name': METRIC_TABLE_NAME,
      'test_name_where_clause': '(test_name LIKE @test_name0)',
      'metric_name_where_clause': '(metric_name LIKE @metric_name0 OR metric_name LIKE @metric_name1)',
      'cutoff_date': (
          datetime.datetime.now() - datetime.timedelta(30)).strftime('%Y-%m-%d')
    }))
    self.assertEqual(len(config['query']['queryParameters']),
                     len(test_names) + len(metric_names))
    self.assertEqual(config['query']['queryParameters'][0], {
      'name': 'test_name0',
      'parameterType': {'type': 'STRING'},
      'parameterValue': {'value': test_names[0]},
    })
    self.assertEqual(config['query']['queryParameters'][1], {
      'name': 'metric_name0',
      'parameterType': {'type': 'STRING'},
      'parameterValue': {'value': metric_names[0]},
    })
    self.assertEqual(config['query']['queryParameters'][2], {
      'name': 'metric_name1',
      'parameterType': {'type': 'STRING'},
      'parameterValue': {'value': metric_names[1]},
    })

  def test_get_query_and_config_multiple_elements(self):
    test_names = ['tf-mnist', 'tf-cifar', 'pt-mnist-%']
    metric_names = ['acc', 'loss']
    query = metric_compare.get_query(test_names, metric_names)
    config = metric_compare.get_query_config(test_names, metric_names)
    self.assertEqual(query, metric_compare.QUERY.format(**{
      'job_table_name': JOB_TABLE_NAME,
      'metric_table_name': METRIC_TABLE_NAME,
      'test_name_where_clause': '(test_name LIKE @test_name0 OR test_name LIKE @test_name1 OR test_name LIKE @test_name2)',
      'metric_name_where_clause': '(metric_name LIKE @metric_name0 OR metric_name LIKE @metric_name1)',
      'cutoff_date': (
          datetime.datetime.now() - datetime.timedelta(30)).strftime('%Y-%m-%d')
    }))
    self.assertEqual(len(config['query']['queryParameters']),
                     len(test_names)+ len(metric_names))
    for i, name in enumerate(test_names):
      self.assertEqual(config['query']['queryParameters'][i], {
        'name': f'test_name{i}',
        'parameterType': {'type': 'STRING'},
        'parameterValue': {'value': test_names[i]},
      })
    self.assertEqual(config['query']['queryParameters'][3], {
      'name': 'metric_name0',
      'parameterType': {'type': 'STRING'},
      'parameterValue': {'value': metric_names[0]},
    })
    self.assertEqual(config['query']['queryParameters'][4], {
      'name': 'metric_name1',
      'parameterType': {'type': 'STRING'},
      'parameterValue': {'value': metric_names[1]},
    })

  def test_make_html_table(self):
    data_grid = [[1,2,'header'], [4,'-',6]]
    self.assertEqual(
        metric_compare.make_html_table(data_grid),
        '<table style="width:300px"><tr><th style="width:100px; border:1px solid #cfcfcf">1</th><th style="width:100px; border:1px solid #cfcfcf">2</th><th style="width:100px; border:1px solid #cfcfcf">header</th></tr><tr><td style="width:100px; border:1px solid #cfcfcf">4</td><td style="width:100px; border:1px solid #cfcfcf">-</td><td style="width:100px; border:1px solid #cfcfcf">6</td></tr></table>')

if __name__ == '__main__':
  absltest.main()
