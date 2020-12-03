# Lint as: python3
"""Tests for main_heatmap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import main_heatmap
import numpy as np
import pandas as pd

SAMPLE_LOGS_LINK = 'https://console.cloud.google.com/logs?project=xl-ml-test&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dxl-ml-test%0Aresource.labels.location=us-central1-b%0Aresource.labels.cluster_name=xl-ml-test%0Aresource.labels.namespace_name=automated%0Aresource.labels.pod_name:pt-1.5-cpp-ops-func-v2-8-1587398400&dateRangeUnbound=backwardInTime'

def _get_values_for_failures(values, statuses):
  return [zipped[0] for zipped in zip(
      values, statuses) if zipped[1] == 'failure']


class MainHeatmapTest(parameterized.TestCase):

  @parameterized.named_parameters(
    ('all_success_all_oob', {
        'job_statuses': ['success', 'success', 'success'],
        'metric_statuses': ['failure', 'failure', 'failure'],
        'expected_overall_statuses': ['failure', 'failure', 'failure'],
        'expected_job_status_abbrevs': ['M', 'M', 'M']}),
    ('all_success_some_oob', {
        'job_statuses': ['success', 'success', 'success'],
        'metric_statuses': ['failure', 'failure', 'success'],
        'expected_overall_statuses': ['failure', 'failure', 'success'],
        'expected_job_status_abbrevs': ['M', 'M', '']}),
    ('all_success_none_oob', {
        'job_statuses': ['success', 'success', 'success'],
        'metric_statuses': ['success', 'success', 'success'],
        'expected_overall_statuses': ['success', 'success', 'success'],
        'expected_job_status_abbrevs': ['', '', '']}),
    ('some_success_some_oob', {
        'job_statuses': ['success', 'failure', 'success'],
        'metric_statuses': ['success', 'success', 'failure'],
        'expected_overall_statuses': ['success', 'failure', 'failure'],
        'expected_job_status_abbrevs': ['', 'F', 'M']}),
  )
  def test_process_dataframes(self, args_dict):
    job_statuses = args_dict['job_statuses']
    metric_statuses = args_dict['metric_statuses']
    assert len(job_statuses) == len(metric_statuses)
    job_status_df = pd.DataFrame({
        'test_name': pd.Series(['test{}'.format(n) for n in range(
            len(job_statuses))]),
        'run_date': pd.Series(['2020-04-{:02d}'.format(n) for n in range(
            len(job_statuses))]),
        'job_status': pd.Series(job_statuses),
        'logs_link': pd.Series([SAMPLE_LOGS_LINK for _ in job_statuses]),
        'logs_download_command': pd.Series(
            ['my command'] + ['' for _ in job_statuses[1:]]),
    })

    # The SQL query in the real code only returns rows where metrics were
    # out of bounds. These oobs rows correspond to 'failure' for
    # metric_statuses in this test.
    metric_names = ['acc' if n % 2 else 'loss' for n in range(
        len(job_status_df))]
    metric_values = [98.0 if n % 2 else 0.6 for n in range(
        len(job_status_df))]
    metric_upper_bounds = [np.nan if n % 2 else 0.5 for n in range(
        len(job_status_df))]
    metric_lower_bounds = [99.0 if n % 2 else np.nan for n in range(
        len(job_status_df))]
    metric_status_df = pd.DataFrame({
        'test_name': pd.Series(_get_values_for_failures(
            job_status_df['test_name'].tolist(), metric_statuses)),
        'run_date': pd.Series(_get_values_for_failures(
            job_status_df['run_date'].tolist(), metric_statuses)),
        'metric_name': pd.Series(_get_values_for_failures(
            metric_names, metric_statuses)),
        'metric_value': pd.Series(_get_values_for_failures(
            metric_values, metric_statuses)),
        'metric_upper_bound': pd.Series(_get_values_for_failures(
            metric_upper_bounds, metric_statuses)),
        'metric_lower_bound': pd.Series(_get_values_for_failures(
            metric_lower_bounds, metric_statuses)),
    })

    # Process the dataframes and make sure the overall_status matches
    # the expected overall_status.
    df = main_heatmap.process_dataframes(job_status_df, metric_status_df)
    self.assertEqual(df['overall_status'].tolist(),
                     args_dict['expected_overall_statuses'])

    self.assertEqual(df['job_status_abbrev'].tolist(),
                     args_dict['expected_job_status_abbrevs'])

    # We only want to display metrics as a top-level failure if the job
    # succeeded. For failed jobs, it's not so helpful to know that the
    # metrics were out of bounds.
    metrics_failure_explanations = df['failed_metrics'].tolist()
    for i, expl_list in enumerate(metrics_failure_explanations):
      job_status = job_statuses[i]
      metric_status = metric_statuses[i]
      if job_status == 'success' and metric_status == 'failure':
        self.assertGreaterEqual(len(expl_list), 1)
        for expl in expl_list:
          self.assertTrue('outside' in expl)
      else:
        self.assertFalse(expl_list)

    commands = df['logs_download_command'].tolist()
    # If the command is already populated, it should be left alone.
    self.assertEqual(commands[0], 'my command')

  def test_process_dataframes_no_job_status(self):
    job_status_df = pd.DataFrame({
        'test_name': pd.Series(['a', 'b']),
        'run_date': pd.Series(['2020-04-10', '2020-04-11']),
        'logs_link': pd.Series(['c', 'd']),
        'logs_download_command': pd.Series(['e', 'f']),
    })
    df = main_heatmap.process_dataframes(job_status_df, pd.DataFrame())
    self.assertTrue(df.empty)
    df = main_heatmap.process_dataframes(pd.DataFrame(), pd.DataFrame())
    self.assertTrue(df.empty)

  def test_make_plot(self):
    input_df = pd.DataFrame({
        'test_name': pd.Series(['test1', 'test2', 'test3']),
        'run_date': pd.Series(['2020-04-21', '2020-04-20', '2020-04-19']),
        'job_status': pd.Series(['success', 'success', 'failure']),
        'logs_link': pd.Series([SAMPLE_LOGS_LINK] * 3),
        'job_status_abbrev': pd.Series(['f', 'f', '']),
        'overall_status': pd.Series(['failure', 'success', 'failure']),
    })
    # Make sure nothing crashes and we are able generate some kind of plot.
    plot = main_heatmap.make_plot(input_df)
    self.assertTrue(plot is not None and len(plot.renderers) > 0)

  def test_make_plot_empty_data(self):
    input_df = pd.DataFrame()
    # Make sure nothing crashes.
    plot = main_heatmap.make_plot(input_df)


if __name__ == '__main__':
  absltest.main()
