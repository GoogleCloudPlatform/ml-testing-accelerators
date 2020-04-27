# Lint as: python3
"""Tests for main_heatmap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import metric_history
import numpy as np
import pandas as pd

SAMPLE_LOGS_LINK = 'https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-cpp-ops-func-v2-8-1587398400&dateRangeUnbound=backwardInTime'


class MetricHistoryTest(parameterized.TestCase):

  def test_process_dataframe(self):
    input_df = pd.DataFrame({
        'logs_link': pd.Series([SAMPLE_LOGS_LINK] * 3),
        'logs_download_command': pd.Series(['no_op', '', 'no_op']),
    })
    df = metric_history.process_dataframe(input_df)
    self.assertEqual(len(df), 3)
    download_commands = df['logs_download_command'].tolist()
    self.assertTrue('gcloud' in download_commands[1])
    self.assertEqual(download_commands[0], 'no_op')
    self.assertEqual(download_commands[2], 'no_op')

  def test_make_plots_nothing_oob(self):
    input_df = pd.DataFrame({
        'test_name': pd.Series(['test1', 'test1', 'test1', 'test1']),
        'metric_name': pd.Series(['acc', 'loss', 'acc', 'loss']),
        'run_date': pd.Series(['2020-04-21', '2020-04-20', '2020-04-20',
                               '2020-04-21']),
        'metric_value': pd.Series([99.1, 0.5, 99.2, 0.6]),
        'metric_upper_bound': pd.Series([np.nan, 1.0, np.nan, 1.0]),
        'metric_lower_bound': pd.Series([99.0, np.nan, 99.0, np.nan]),
        'logs_link': pd.Series([SAMPLE_LOGS_LINK] * 4),
        'job_status': pd.Series(['success', 'success', 'success', 'success']),
    })
    # There should be 2 plots: 1 per metric. Neither should be outlined in red
    # since neither metric was oob.
    plots = metric_history.make_plots('test1', '', input_df)
    self.assertEqual(len(plots), 2)
    self.assertItemsEqual([plot.title.text for plot in plots], ['loss', 'acc'])
    self.assertNotEqual(plots[0].outline_line_color, 'red')
    self.assertNotEqual(plots[1].outline_line_color, 'red')

  def test_make_plots_with_oob(self):
    input_df = pd.DataFrame({
        'test_name': pd.Series(['test1', 'test1', 'test1', 'test1']),
        'metric_name': pd.Series(['acc', 'loss', 'acc', 'loss']),
        'run_date': pd.Series(['2020-04-21', '2020-04-20', '2020-04-20',
                               '2020-04-21']),
        'metric_value': pd.Series([98.1, 0.5, 99.2, 0.6]),
        'metric_upper_bound': pd.Series([np.nan, 1.0, np.nan, 1.0]),
        'metric_lower_bound': pd.Series([99.0, np.nan, 99.0, np.nan]),
        'logs_link': pd.Series([SAMPLE_LOGS_LINK] * 4),
        'job_status': pd.Series(['success', 'success', 'success', 'success']),
    })
    # There should be 2 plots: 1 per metric.
    plots = metric_history.make_plots('test1', '', input_df)
    self.assertEqual(len(plots), 2)
    # 'acc' should come first since it is oob. It should be outlined in red.
    self.assertEqual([plot.title.text for plot in plots], ['acc', 'loss'])
    self.assertEqual(plots[0].outline_line_color, 'red')
    self.assertNotEqual(plots[1].outline_line_color, 'red')

  def test_make_plots_with_oob_on_old_date(self):
    input_df = pd.DataFrame({
        'test_name': pd.Series(['test1', 'test1', 'test1', 'test1']),
        'metric_name': pd.Series(['acc', 'loss', 'acc', 'loss']),
        'run_date': pd.Series(['2020-04-21', '2020-04-20', '2020-04-20',
                               '2020-04-21']),
        'metric_value': pd.Series([99.1, 0.5, 98.2, 0.6]),
        'metric_upper_bound': pd.Series([np.nan, 1.0, np.nan, 1.0]),
        'metric_lower_bound': pd.Series([99.0, np.nan, 99.0, np.nan]),
        'logs_link': pd.Series([SAMPLE_LOGS_LINK] * 4),
        'job_status': pd.Series(['success', 'success', 'success', 'success']),
    })
    # There should be 2 plots: 1 per metric.
    plots = metric_history.make_plots('test1', '', input_df)
    self.assertEqual(len(plots), 2)
    # 'acc' was oob 2 runs ago but most recent run was OK, so it should not
    # be given a red outline.
    self.assertItemsEqual([plot.title.text for plot in plots], ['acc', 'loss'])
    self.assertNotEqual(plots[0].outline_line_color, 'red')
    self.assertNotEqual(plots[1].outline_line_color, 'red')

  def test_make_plots_with_metric_substr(self):
    input_df = pd.DataFrame({
        'test_name': pd.Series(['test1', 'test1', 'test1', 'test1']),
        'metric_name': pd.Series(['acc', 'loss', 'acc', 'loss']),
        'run_date': pd.Series(['2020-04-21', '2020-04-20', '2020-04-20',
                               '2020-04-21']),
        'metric_value': pd.Series([99.1, 0.5, 98.2, 0.6]),
        'metric_upper_bound': pd.Series([np.nan, 1.0, np.nan, 1.0]),
        'metric_lower_bound': pd.Series([99.0, np.nan, 99.0, np.nan]),
        'logs_link': pd.Series([SAMPLE_LOGS_LINK] * 4),
        'job_status': pd.Series(['success', 'success', 'success', 'success']),
    })
    # There should be only 1 plot since we're using 'loss' as search substr.
    plots = metric_history.make_plots('test1', 'loss', input_df)
    self.assertEqual(len(plots), 1)
    self.assertItemsEqual([plot.title.text for plot in plots], ['loss'])
    self.assertNotEqual(plots[0].outline_line_color, 'red')


if __name__ == '__main__':
  absltest.main()
