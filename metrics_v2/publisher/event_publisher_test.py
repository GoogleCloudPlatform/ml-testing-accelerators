import collections
import datetime
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import kubernetes

import event_publisher
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import json_format
import metrics_pb2

_START_TIME = datetime.datetime.fromisoformat('2020-12-15T19:58:44')
_END_TIME = datetime.datetime.fromisoformat('2020-12-15T20:14:56')

# HACK: See https://github.com/kubernetes-client/python/issues/977#issuecomment-594045477
def _job_from_dict(d):
  _FakeResponse = collections.namedtuple('FakeResponse', 'data')
  resp = _FakeResponse(json.dumps(d, default=str))
  return kubernetes.client.BatchV1Api().api_client.deserialize(resp, 'V1Job')

class EventPublisherTest(parameterized.TestCase):
  def assertProtoEqual(self, first, second):
    self.assertJsonEqual(
      json_format.MessageToJson(first, including_default_value_fields=True),
      json_format.MessageToJson(second, including_default_value_fields=True),
    )

  @parameterized.named_parameters(
    ('passing', 1, 0, [('Complete', None)], 'COMPLETED'),
    ('retried_passing', 1, 1, [('Complete', None)], 'COMPLETED'),
    ('failing', 0, 2, [('Failed', 'BackoffLimitExceeded')], 'FAILED'),
    ('timed_out', 0, 1, [('Failed', 'DeadlineExceeded')], 'TIMEOUT'),
    ('barely_timed_out', 0, 1, [('Failed', 'DeadlineExceeded'), ('Complete', None)], 'TIMEOUT'),
  )
  def test_create_test_completed_event(self, succeeded_count, failed_count, conditions, expected_status):
    job = _job_from_dict({
      'metadata': {
        'name': 'job-name',
        'namespace': 'namespace',
        'labels': {
          'benchmarkId': 'test-job',
        },
      },
      'status': {
        'startTime': _START_TIME,
        'completionTime': _END_TIME,
        'succeeded': succeeded_count,
        'failed': failed_count,
        'conditions': [
          {
            'status': True,
            'reason': reason,
            'type': cond_type,
          }
          for cond_type, reason in conditions
        ]
      }
    })

    actual_event = event_publisher.create_test_completed_event(
      job,
      model_output_bucket='gs://fake-bucket',
      cluster_name='cluster-name',
      cluster_location='cluster-location',
      project='project-id'
    )

    start_time = timestamp_pb2.Timestamp()
    start_time.FromDatetime(_START_TIME)
    duration = duration_pb2.Duration()
    duration.FromTimedelta(_END_TIME - _START_TIME)
    expected_event = metrics_pb2.TestCompletedEvent(
      benchmark_id='test-job',
      output_path='gs://fake-bucket/job-name',
      status=metrics_pb2.TestCompletedEvent.TestStatus.Value(expected_status),
      num_attempts=succeeded_count + failed_count,
      start_time=start_time,
      duration=duration,
      labels={'benchmarkId': 'test-job'},
      debug_info=metrics_pb2.DebugInfo(
        logs_link='https://console.cloud.google.com/logs?project=project-id&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3Dproject-id%0Aresource.labels.cluster_name%3D%24cluster-name%0Aresource.labels.namespace_name%3Dnamespace%0Aresource.labels.pod_name%3Ajob-name%0Aresource.labels.location%3Acluster-location%0A',
        details_link=f'https://console.cloud.google.com/kubernetes/job/cluster-location/cluster-name/namespace/job-name?project=project-id'
      ),
      metric_collection_config=metrics_pb2.MetricCollectionConfig(),
    )

    self.assertProtoEqual(expected_event, actual_event)

  @parameterized.named_parameters(
    ('with_subdir', 'some/subdir/path'),
    ('no_subdir', None),
  )
  def test_metric_collection_config(self, gcs_subdir):
    job = _job_from_dict({
      'metadata': {
        'name': 'job-name',
        'namespace': 'namespace',
        'labels': {
          'benchmarkId': 'test-job',
        },
        'annotations': {
          'ml-testing-accelerators/metric-config': json.dumps({
            'sources':  [{
              'literals': {
                'assertions': {
                  'duration': {
                    'within_bounds': {
                      'lower_bound': 1,
                      'upper_bound': 2,
                    }
                  }
                }
              }
            }]
          })
        }
      },
      'status': {
        'startTime': _START_TIME,
        'completionTime': _END_TIME,
        'succeeded': 1,
        'conditions': [
          {
            'status': True,
            'type': 'Complete',
          }
        ]
      }
    })
    if gcs_subdir:
      job.metadata.annotations['ml-testing-accelerators/gcs-subdir'] = gcs_subdir

    actual_event = event_publisher.create_test_completed_event(
      job,
      model_output_bucket='gs://fake-bucket',
      cluster_name='cluster-name',
      cluster_location='cluster-location',
      project='project-id'
    )
    actual_mcc = actual_event.metric_collection_config

    expected_mcc = metrics_pb2.MetricCollectionConfig(
      sources=[
        metrics_pb2.MetricSource(
          literals=metrics_pb2.LiteralSource(
            assertions={
              'duration': metrics_pb2.Assertion(
                within_bounds=metrics_pb2.Assertion.WithinBounds(
                  lower_bound=1,
                  upper_bound=2,
                )
              )
            }
          )
        )
      ]
    )
    self.assertEqual(actual_event.output_path, os.path.join('gs://fake-bucket', gcs_subdir or '', 'job-name'))
    self.assertProtoEqual(expected_mcc, actual_mcc)

if __name__ == "__main__":
  absltest.main()

