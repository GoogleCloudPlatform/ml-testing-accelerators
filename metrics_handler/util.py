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

# Lint as: python3
"""Util methods used by the Cloud Accelerators Testing metrics handler."""

import math
import re


LOGS_DOWNLOAD_COMMAND = """gcloud logging read 'resource.type=k8s_container resource.labels.project_id={project} resource.labels.location={zone} resource.labels.cluster_name={cluster} resource.labels.namespace_name={namespace} resource.labels.pod_name:{pod}' --limit 10000000000000 --order asc --format 'value(textPayload)' --project=xl-ml-test > {pod}_logs.txt && sed -i '/^$/d' {pod}_logs.txt"""
LOG_LINK_REGEX = re.compile('https://console\.cloud\.google\.com/logs\?project=(\S+)\&advancedFilter=resource\.type\%3Dk8s_container\%0Aresource\.labels\.project_id\%3D(?P<project>\S+)\%0Aresource\.labels\.location=(?P<zone>\S+)\%0Aresource\.labels\.cluster_name=(?P<cluster>\S+)\%0Aresource\.labels\.namespace_name=(?P<namespace>\S+)\%0Aresource\.labels\.pod_name:(?P<pod>[a-zA-Z0-9\-\.]+)')
UNBOUND_DATE_RANGE = '&dateRangeUnbound=backwardInTime'

def _parse_logs_link(logs_link):
  log_pieces = LOG_LINK_REGEX.match(logs_link)
  if not log_pieces:
    raise ValueError('Could not parse Stackdriver logs link. '
                     'Logs link was: {}'.format(logs_link))
  return log_pieces.groupdict()


def add_unbound_time_to_logs_link(logs_link):
  """Add dateRangeUnbound arg to `logs_link` if it doesn't already exist.

  Args:
    logs_link (string): Link to the logs of a Kubernetes pod.

  Returns:
    logs_link (string): Input with dateRangeUnbound added to it or the
      input unchanged if it already contained dateRangeUnbound.
  """
  return logs_link if UNBOUND_DATE_RANGE in logs_link else \
      logs_link + UNBOUND_DATE_RANGE


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


def replace_invalid_values(row):
  """Replace float values that are not available in BigQuery.

  Args:
    row: List of values to insert into BigQuery.

  Returns:
    List, `row` with invalid values replaced with `None`.
  """
  invalid_values = [math.inf, -math.inf, math.nan]
  return [x if x not in invalid_values else None for x in row]
