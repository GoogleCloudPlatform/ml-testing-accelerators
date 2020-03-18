# Lint as: python3
"""Util methods used by the Cloud Accelerators Testing metrics handler."""

import re


LOGS_DOWNLOAD_COMMAND = """gcloud logging read 'resource.type=k8s_container resource.labels.project_id={project} resource.labels.location={zone} resource.labels.cluster_name={cluster} resource.labels.namespace_name={namespace} resource.labels.pod_name:{pod}' --limit 10000000000000 --order asc --format 'value(textPayload)' > {pod}_logs.txt && sed -i '/^$/d' {pod}_logs.txt"""
LOG_LINK_REGEX = re.compile('https://console\.cloud\.google\.com/logs\?project=(\S+)\&advancedFilter=resource\.type\%3Dk8s_container\%0Aresource\.labels\.project_id\%3D(?P<project>\S+)\%0Aresource\.labels\.location=(?P<zone>\S+)\%0Aresource\.labels\.cluster_name=(?P<cluster>\S+)\%0Aresource\.labels\.namespace_name=(?P<namespace>\S+)\%0Aresource\.labels\.pod_name:(?P<pod>[a-zA-Z0-9\-]+)')

def _parse_logs_link(logs_link):
  log_pieces = LOG_LINK_REGEX.match(logs_link)
  if not log_pieces:
    raise ValueError('Could not parse Stackdriver logs link. '
                     'Logs link was: {}'.format(logs_link))
  return log_pieces.groupdict()


def download_command_from_logs_link(logs_link):
  """Convert a link to Stackdriver logs to a command to download logs.

  Args:
    logs_link (string): Link to the logs of a Kubernetes pod.

  Returns:
    command (string): A command to download the plaintext logs of the pod
      that was referenced in `logs_link`.
  """
  log_pieces_dict = _parse_logs_link(logs_link)
  command = LOGS_DOWNLOAD_COMMAND.format(**log_pieces_dict)
  return command


def test_name_from_logs_link(logs_link):
  """Extract test name from a link to Stackdriver logs for that test.

  Args:
    logs_link (string): Link to the logs of a Kubernetes pod.

  Returns:
    test_name (string): The name of the Kuberneted pod, which is synonymous
      with the test name in the context of the metrics handler.
  """
  log_pieces_dict = _parse_logs_link(logs_link)
  test_name = log_pieces_dict.get('pod')
  if not test_name:
    raise ValueError('Unable to parse test name from logs link: {}'.format(
        logs_link))
  return test_name
