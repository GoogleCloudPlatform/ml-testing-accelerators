# Lint as: python3
"""Tests for main_heatmap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import main_heatmap
import numpy as np
import pandas as pd

class MainHeatmapTest(absltest.TestCase):

  def test_process_dataframes_all_pass_within_bounds(self):
    #process_dataframes(job_status_dataframe, metrics_dataframe)
    job_status_df = pd.DataFrame({'test_name': pd.Series(['test1', 'test2', 'test3']), 'job_status': pd.Series(['success', 'success', 'success'])})
    metric_status_df = pd.DataFrame({
        'test_name': pd.Series(['test1', 'test2', 'test3']),
        'run_date': pd.Series(['2020-04-20', '2020-04-20', '2020-04-19']),
        'metric_name': pd.Series(['acc', 'loss', 'acc']),
        'metric_value': pd.Series([99.1, 0.1, 99.2]),
        'metric_upper_bound': pd.Series([np.nan, 0.5, np.nan]),
        'metric_lower_bound': pd.Series([99.0, np.nan, 99.0]),
    })
    df = main_heatmap.process_dataframes(job_status_df, metric_status_df)
    pass


if __name__ == '__main__':
  absltest.main()
