import argparse
import google.api_core.exceptions
from google.cloud import monitoring_v3


def cleanup_alert_policies(test_name):
  """Delete all alert policies whose display name contains `test_name`."""
  alert_client = monitoring_v3.AlertPolicyServiceClient()
  project_name = alert_client.project_path(google.auth.default()[1])

  # First find the unique ID for all the existing policies.
  # The ID is required to update or delete existing policies.
  num_policies_deleted = 0
  deleted_policies = []
  for p in alert_client.list_alert_policies(project_name):
    if p.display_name.find(test_name) >= 0:
      alert_client.delete_alert_policy(p.name)
      num_policies_deleted += 1
      deleted_policies.append(p.display_name)
  print('Deleted policies: {}'.format(deleted_policies))
  print('Number of policies deleted: {}'.format(num_policies_deleted))


def cleanup_metrics(test_name):
  metric_client = monitoring_v3.MetricServiceClient()
  project_name = metric_client.project_path(google.auth.default()[1])
  num_metrics_deleted = 0
  deleted_metrics = []
  for x in metric_client.list_metric_descriptors(project_name):
    if x.name.find(test_name) >= 0:
      metric_client.delete_metric_descriptor(x.name)
      num_metrics_deleted += 1
      deleted_metrics.append(x.name)
  print('Deleted metrics: {}'.format(deleted_metrics))
  print('Number of metrics deleted: {}'.format(num_metrics_deleted))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
          description='Batch delete of Stackdriver Alert Policies.',
          epilog=('Usage example: cleanup.py '
              '--test_name="pt-nightly-resnet50" '))
  parser.add_argument('--type', type=str, default=None,
                      help='Type of resource to clean up, e.g. "alerts" or '
                      '"metrics".', required=True)
  parser.add_argument('--test_name', type=str, default=None,
                      help='Delete all alert policies that contain this '
                      'substring.', required=True)
  FLAGS = parser.parse_args()
  if FLAGS.type == 'alerts':
    cleanup_alert_policies(FLAGS.test_name)
  elif FLAGS.type == 'metrics':
    cleanup_metrics(FLAGS.test_name)
  else:
    raise ValueError('Unrecognized "type" argument: {}'.format(FLAGS.type))

