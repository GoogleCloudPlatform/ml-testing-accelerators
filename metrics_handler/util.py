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


LOGS_DOWNLOAD_COMMAND = """gcloud logging read 'resource.type=k8s_container resource.labels.project_id={project} resource.labels.location={zone} resource.labels.cluster_name={cluster} resource.labels.namespace_name={namespace} resource.labels.pod_name:{pod}' --limit 10000000000000 --order asc --format 'value(textPayload)' --project={project} > {pod}_logs.txt && sed -i '/^$/d' {pod}_logs.txt"""
UNBOUND_DATE_RANGE = '&dateRangeUnbound=backwardInTime'
WORKLOAD_LINK = """https://console.cloud.google.com/kubernetes/job/{location}/{cluster}/{namespace}/{job}?project={project}"""


def add_unbound_time_to_logs_link(logs_link):
  """Add dateRangeUnbound arg to `logs_link` if it doesn't already exist.

  Args:
    logs_link (string): Link to the logs of a Kubernetes pod.

  Returns:
    logs_link (string): Input with dateRangeUnbound added to it or the
      input unchanged if it already contained dateRangeUnbound.
  """
  return logs_link if (not logs_link or UNBOUND_DATE_RANGE in logs_link) else \
      logs_link + UNBOUND_DATE_RANGE


def download_command(job_name, job_namespace, zone, cluster, project):
  """Build a command to download Stackdriver logs for a test run.

  Args:
    job_name (string): Name of the Kubernetes job. Should include the
      timestamp, e.g. 'pt-1.5-resnet-func-v3-8-1584453600'.
    job_namespace (string): Name of the Kubernetes namespace.
    zone (string): GCP zone, e.g. 'us-central1-b'.
    cluster (string): Name of the Kubernetes cluster.
    project (string): Name of the GCP project.

  Returns:
    command (string): A command to download the plaintext logs of the pod
      that was referenced in `logs_link`.
  """
  command = LOGS_DOWNLOAD_COMMAND.format(**{
      'project': project,
      'zone': zone,
      'namespace': job_namespace,
      'pod': job_name,
      'cluster': cluster
  })
  return command


def workload_link(job_name, job_namespace, location, cluster, project):
  """Build a link to the Kubernetes workload for a specific test run.

  Args:
    job_name (string): Name of the Kubernetes job. Should include the
      timestamp, e.g. 'pt-1.5-resnet-func-v3-8-1584453600'.
    job_namespace (string): Name of the Kubernetes namespace.
    location (string): GCP zone or region, e.g. 'us-central1-b'.
    cluster (string): Name of the Kubernetes cluster.
    project (string): Name of the GCP project.

  Returns:
    link (string): A link to the Kubernetes workload page for this job.
  """
  link = WORKLOAD_LINK.format(**{
      'project': project,
      'location': location,
      'namespace': job_namespace,
      'job': job_name,
      'cluster': cluster
  })
  return link


def is_valid_bigquery_value(v):
  """Return True if value is valid for writing to BigQuery.

  Args:
    v (float): Value to check.

  Returns:
    Bool, True if v is valid and False otherwise.

  """
  invalid_values = [math.inf, -math.inf, math.nan]
  if v in invalid_values:
    return False
  if math.isnan(v):
    return False
  return True


def replace_invalid_values(row):
  """Replace float values that are not available in BigQuery.

  Args:
    row: List of values to insert into BigQuery.

  Returns:
    List, `row` with invalid values replaced with `None`.
  """
  return [x if is_valid_bigquery_value(x) else None for x in row]
