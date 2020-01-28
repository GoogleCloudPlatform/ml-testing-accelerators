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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
          description='Batch delete of Stackdriver Alert Policies.',
          epilog=('Usage example: cleanup.py '
              '--test_name="pt-nightly-resnet50" '))
  parser.add_argument('--test_name', type=str, default=None,
                      help='Delete all alert policies that contain this '
                      'substring.', required=True)
  FLAGS = parser.parse_args()
  cleanup_alert_policies(FLAGS.test_name)

