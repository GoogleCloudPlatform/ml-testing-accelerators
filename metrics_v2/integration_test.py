import collections
import contextlib
import dataclasses
import datetime
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import kubernetes
import tensorflow as tf


os.environ['BQ_DATASET'] = 'fake-dataset'
import handler.main
import handler.bigquery_client
from publisher import event_publisher
import metrics_pb2

# HACK: See https://github.com/kubernetes-client/python/issues/977#issuecomment-594045477
def _job_from_dict(d):
  _FakeResponse = collections.namedtuple('FakeResponse', 'data')
  resp = _FakeResponse(json.dumps(d, default=str))
  return kubernetes.client.BatchV1Api().api_client.deserialize(resp, 'V1Job')

class IntegrationTest(parameterized.TestCase):
  def setUp(self):
    self.job_name = 'job-name'
    self.temp_dir = self.create_tempdir().full_path
    summary_writer = tf.summary.create_file_writer(
        os.path.join(self.temp_dir, self.job_name))
    with summary_writer.as_default(), contextlib.closing(summary_writer):
      tf.summary.scalar("accuracy", .1, 0)
      tf.summary.scalar("accuracy", .75, 100)

  def test_publisher_handler(self):
    time_1 = datetime.datetime.fromisoformat('2020-12-15T19:58:44')
    time_2 = datetime.datetime.fromisoformat('2020-12-15T20:14:56')
    job = _job_from_dict({
      'metadata': {
        'name': self.job_name,
        'namespace': 'namespace',
        'labels': {
          'benchmarkId': 'test-job',
          'model': 'test-model',
          'mode': 'test-mode',
          'accelerator': 'test-accelerator',
          'frameworkVersion': 'test-framework',
        },
        'annotations': {
          'ml-testing-accelerators/metric-config': json.dumps({
            'sources':  [
              {
                'literals': {
                  # This assertion should be OOB.
                  'assertions': {
                    'duration': {
                      'within_bounds': {
                        'lower_bound': 1,
                        'upper_bound': 2,
                      }
                    }
                  }
                }
              },
              {
                'tensorboard': {
                  # This assertion should be within-bounds.
                  'aggregate_assertions': [{
                    'tag': 'accuracy',
                    'strategy': 'FINAL',
                    'assertion': {
                      'within_bounds': {
                        'lower_bound': .7,
                        'upper_bound': 1.0,
                      }
                    }
                  }]
                },
              }
            ]
          })
        }
      },
      'status': {
        'startTime': time_1,
        'completionTime': time_2,
        'succeeded': 1,
        'failed': 1,
        'conditions': [
          {
            'status': True,
            'type': 'Complete',
          }
        ]
      }
    })
    event = event_publisher.create_test_completed_event(
      job,
      model_output_bucket=self.temp_dir,
      cluster_name='cluster-name',
      cluster_location='cluster-location',
      project='project-id',
    )

    with self.assertLogs() as cm:
      job_row, metric_rows = handler.main.process_proto_message(event, None)
      self.assertTrue(any('duration' in line for line in cm.output if line.startswith('ERROR')), 'Error log expected for metric `duration`. Logs: {}'.format(cm.output))
      self.assertFalse(any('accuracy' in line for line in cm.output if line.startswith('ERROR')), 'Error log not expected for metric `accuracy`. Logs: {}'.format(cm.output))

    uuid = job_row.uuid
    for row in metric_rows:
      self.assertEqual(uuid, row.uuid)

    duration = time_2 - time_1
    self.assertDictEqual(
        dataclasses.asdict(job_row), 
        dataclasses.asdict(handler.bigquery_client.JobHistoryRow(
            uuid,
            test_name='test-job',
            test_type='test-mode',
            accelerator='test-accelerator',
            framework_version='test-framework',
            job_status='success',
            num_failures=1,
            job_duration_sec=duration.seconds,
            timestamp=time_1,
            stackdriver_logs_link='https://console.cloud.google.com/logs?project=project-id&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dproject-id%0Aresource.labels.cluster_name%3D%24cluster-name%0Aresource.labels.namespace_name%3Dnamespace%0Aresource.labels.pod_name%3Ajob-name%0Aresource.labels.location%3Acluster-location%0A',
            msg_publish_time=time_1.timestamp() + duration.total_seconds(),
            kubernetes_workload_link=f'https://console.cloud.google.com/kubernetes/job/cluster-location/cluster-name/namespace/job-name?project=project-id',
            logs_download_command='',
        ))
    )

    self.assertCountEqual(
      [dataclasses.asdict(row) for row in metric_rows], 
      [
          dataclasses.asdict(handler.bigquery_client.MetricHistoryRow(
              uuid,
              test_name='test-job',
              timestamp=time_1,
              metric_value=.75,
              metric_name='accuracy_final',
              metric_lower_bound=.7,
              metric_upper_bound=1.,
          )),
          dataclasses.asdict(handler.bigquery_client.MetricHistoryRow(
              uuid,
              test_name='test-job',
              timestamp=time_1,
              metric_name='duration',
              metric_value=duration.total_seconds(),
              metric_lower_bound=1.,
              metric_upper_bound=2.,
          )),
      ]
  )
  

if __name__ == '__main__':
  absltest.main()
