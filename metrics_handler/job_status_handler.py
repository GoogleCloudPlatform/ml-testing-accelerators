# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from google.cloud import container_v1
import google.auth
import google.auth.transport.requests
import kubernetes
from tempfile import NamedTemporaryFile
import time

DOES_NOT_EXIST = 'does_not_exist'
UNKNOWN_STATUS = 'unknown'
SUCCESS = 'success'
FAILURE = 'failure'
TIMEOUT = 'timeout'


class JobStatusHandler(object):
  def __init__(self, project_id, zone, cluster_name, logger):
    """Query Kubernetes API to retrieve the status of Kubernetes Jobs.

    Args:
      project_id (string): Name of the Cloud project under which the
          Kubernetes Job(s) ran.
      zone (string): Zone where the Kubernetes Job(s) ran.
      cluster_name (string): Name of the Kubernetes cluster where the
          Job(s) ran. Corresponds to the `cluster-name` attribute in GCP.
      logger (`AlertHandler` instance): Used to write logs and alert emails.

    Raises:
      Exception if unable to initialize the Kubernetes client.
    """
    self.logger = logger
    # Attempt to initialize a Kubernetes client to retrieve Job statuses.
    # Different methods are used depending on where this code runs.
    try:
      # This method is used when there is no local kubeconfig file, e.g.
      # running this code within a Cloud Function. For local runs, you can
      # use this path by running `gcloud auth application-default login`.
      self.logger.info('Attempting to init k8s client from cluster response.')
      container_client = container_v1.ClusterManagerClient()
      response = container_client.get_cluster(project_id, zone, cluster_name)
      credentials, project = google.auth.default(
          scopes=['https://www.googleapis.com/auth/cloud-platform'])
      creds, projects = google.auth.default()
      auth_req = google.auth.transport.requests.Request()
      creds.refresh(auth_req)
      configuration = kubernetes.client.Configuration()
      configuration.host = f'https://{response.endpoint}'
      with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(
            base64.b64decode(response.master_auth.cluster_ca_certificate))
      configuration.ssl_ca_cert = ca_cert.name
      configuration.api_key_prefix['authorization'] = 'Bearer'
      configuration.api_key['authorization'] = creds.token

      self.k8s_client = kubernetes.client.BatchV1Api(
          kubernetes.client.ApiClient(configuration))
      self.logger.info('Successful init of k8s client from cluster response.')
    except Exception as e1:
      # This method is generally used for local runs where the user has already
      # ran `gcloud container clusters get-credentials` to get a kubeconfig.
      self.logger.warning(
          'Failed to load k8s client from cluster response: {}. '
          'Falling back to local kubeconfig file.'.format(e1))
      try:
        kubernetes.config.load_kube_config()
        self.k8s_client = kubernetes.client.BatchV1Api()
        self.logger.info(
            'Successful init of k8s client from local kubeconfig file.')
      except Exception as e2:
        self.logger.fatal(
            'Failed both methods of loading k8s client. Error for '
            'cluster response method: {}.  Error for local '
            'kubeconfig file: {}.  No job status will be '
            'collected.'.format(e1, e2))
        raise


  def get_job_status(self, job_name, namespace):
    """Returns key information about the status of a Kubernetes Job.

    Args:
      job_name (string): Name of the job.
      namespace (string): Name of the Kubernetes namespace where the job ran.

    Returns:
      completion_code (string): The current status of the Job.
      start_time (timestamp): Time at which the Job began.
      stop_time (timestamp): Time at which the Job completed or fully failed.
      num_failures (int): Number of unsuccessful attempts of this Job.
    """
    try:
      status = self.k8s_client.read_namespaced_job_status(
          job_name, namespace).status
    except Exception as e:
      if isinstance(e, kubernetes.client.rest.ApiException) and \
          e.status == 404:
        self.logger.error(
            'Job with job_name: {} no longer exists in namespace: '
            '{}.  Error was: {}'.format(job_name, namespace, e))
        return DOES_NOT_EXIST, None, None, None
      else:
        self.logger.error(
            'Failed to get job status for job_name: {} and '
            'namespace: {}.  Error was: {}'.format(
                job_name, namespace, e))
        return UNKNOWN_STATUS, None, None, None
    self.logger.info('job_name: {}. status: {}'.format(job_name, status))
    start_time = status.start_time.timestamp()
    if status.active:
      self.logger.error('Job is still active. Returning UNKNOWN_STATUS.')
      return UNKNOWN_STATUS, start_time, None, None

    # Interpret status and return the important parts.
    completion_code = UNKNOWN_STATUS
    if status.succeeded:
      completion_code = SUCCESS
      if status.completion_time:
        stop_time = status.completion_time.timestamp()
      else:
        self.logger.error(
            'No completion_time in success status: {}'.format(status))
        if status.conditions and len(status.conditions) == 1 and \
            status.conditions[0].last_transition_time:
          stop_time = status.conditions[0].last_transition_time.timestamp()
        else:
          stop_time = time.time()
    else:
      if len(status.conditions) != 1:
        self.logger.error('Expected exactly 1 `condition` element in status.')
        completion_code = FAILURE
        stop_time = status.start_time.timestamp()
      else:
        completion_code = TIMEOUT if \
            status.conditions[0].reason == 'DeadlineExceeded' else FAILURE
        stop_time = status.conditions[0].last_transition_time.timestamp()

    num_failures = status.failed or 0
    return completion_code, start_time, stop_time, num_failures
