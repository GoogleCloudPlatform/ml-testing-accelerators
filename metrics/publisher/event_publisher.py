import base64
import os
import re
import textwrap
import typing
import urllib.parse

from absl import app
from absl import flags
from absl import logging
import google.auth
from google.cloud import pubsub_v1
from google.protobuf import json_format
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
import kubernetes
import requests
import pathlib

import metrics_pb2

FLAGS = flags.FLAGS

flags.DEFINE_integer('resource_version', None, 'Cluster resourceVersion to start with.')
flags.DEFINE_string('save_dir', None, 'Directory to save resource_version.txt')

flags.DEFINE_string('pubsub_topic', None, 'PubSub topic to publish to.')
flags.DEFINE_string('model_output_bucket', None, 'GCS location of model outputs.')

flags.DEFINE_string('project', None, 'GCP Project ID of cluster and PubSub topic.')
flags.DEFINE_string('cluster_name', None, 'Name of current cluster.')
flags.DEFINE_string('cluster_location', None, 'Location of current cluster.')
flags.DEFINE_string('namespace', 'default', 'Namespace to watch for Jobs.')


def _resource_version_path():
  if FLAGS.save_dir:
    return os.path.join(FLAGS.save_dir, 'resource_version.txt')
  else:
    return 'resource_version.txt'

def _save_resource_version(resource_version: int):
  logging.debug('Saving resource_version %d', resource_version)

  with open(_resource_version_path(), 'w') as f:
    f.write(str(resource_version))

def _load_resource_version() -> typing.Optional[int]:
  try:
    with open(_resource_version_path(), 'r') as f:
      return int(f.read())
  except FileNotFoundError as e:
    logging.warning('Resource version file not found.', exc_info=e)
    return None

def _update_health():
  pathlib.Path('/tmp/health').touch()

def _get_metadata(attribute):
  resp = requests.get(
      f'http://metadata.google.internal/computeMetadata/v1/instance/attributes/{attribute}',
      headers={'Metadata-Flavor': 'Google'})
  resp.raise_for_status()

  return resp.text

def create_test_completed_event(
    job: kubernetes.client.V1Job,
    model_output_bucket: str,
    cluster_name: str,
    cluster_location: str,
    project:str
) -> metrics_pb2.TestCompletedEvent:
  """Returns a TestCompletedEvent to publish to PubSub.

  Args:
    job: A Kubernetes Job resource.
    model_output_bucket: Path to GCS bucket with model outputs.
    cluster_name: Name of the current Kubernetes cluster.
    cluster_location: Location (region or zone) of the current Kubernetes cluster.
    project: The project ID of the current project.

  Returns:
    A TestCompletedEvent with the information from job.
  """
  if len(job.status.conditions) == 1:
    condition = job.status.conditions[0]
  # job.status.conditions _usually_ has length 1, but it can have both passing and failing conditions.
  # Give precedence to failing conditions.
  elif len(job.status.conditions) == 0:
    logging.error('Job %s has no conditions.', job.metadata.name)
    return
  else:
    condition = next((c for c in job.status.conditions if c.type == 'Failed'), None)

  if not condition:
    logging.error('This should never happen. Conditions: %s', str(job.status.conditions))
    return
  elif condition.reason == 'DeadlineExceeded':
    job_status = metrics_pb2.TestCompletedEvent.TIMEOUT
  elif condition.reason == 'BackoffLimitExceeded':
    job_status = metrics_pb2.TestCompletedEvent.FAILED
  elif condition.type == 'Complete':
    job_status = metrics_pb2.TestCompletedEvent.COMPLETED
  else:
    logging.error('Unknown condition for Job %s: %s', job.metadata.name, str(condition))
    return

  annotations = job.metadata.annotations or {}
  gcs_subdir = annotations.get('ml-testing-accelerators/gcs-subdir', '')
  output_path = os.path.join(model_output_bucket, gcs_subdir, job.metadata.name)

  metric_config = metrics_pb2.MetricCollectionConfig()
  mcc_json = annotations.get('ml-testing-accelerators/metric-config', '{}')
  json_format.Parse(mcc_json, metric_config)

  stackdriver_query = textwrap.dedent(f"""\
    resource.type=k8s_container
    resource.labels.project_id={project}
    resource.labels.cluster_name={cluster_name}
    resource.labels.namespace_name={job.metadata.namespace}
    resource.labels.pod_name:{job.metadata.name}
    resource.labels.location:{cluster_location}
  """)
  stackdriver_link = "https://console.cloud.google.com/logs?{}".format(
      urllib.parse.urlencode(
          {'project': project, 'advancedFilter': stackdriver_query}))

  start_time = timestamp_pb2.Timestamp()
  start_time.FromDatetime(job.status.start_time)
  duration = duration_pb2.Duration()
  duration.FromTimedelta(condition.last_transition_time - job.status.start_time)

  return metrics_pb2.TestCompletedEvent(
    benchmark_id=job.metadata.labels['benchmarkId'],
    output_path=output_path,
    status=job_status,
    num_attempts=(job.status.succeeded or 0) + (job.status.failed or 0),
    start_time=start_time,
    duration=duration,
    metric_collection_config=metric_config,
    labels=job.metadata.labels,
    debug_info=metrics_pb2.DebugInfo(
      logs_link=stackdriver_link,
      # TODO: fix hard-coded region and cluster name
      details_link=f'https://console.cloud.google.com/kubernetes/job/{cluster_location}/{cluster_name}/{job.metadata.namespace}/{job.metadata.name}?project={project}'
    )
  )

def main(argv):
  try:
    kubernetes.config.load_incluster_config()
  except:
    logging.warning("No cluster config found. Trying local kube config.")
    kubernetes.config.load_kube_config()

  resource_version = FLAGS.resource_version or _load_resource_version()

  project = FLAGS.project or google.auth.default()[1]
  cluster_name = FLAGS.cluster_name or _get_metadata('cluster-name')
  cluster_location = FLAGS.cluster_location or _get_metadata('cluster-location')

  if re.match('^projects/[^/]+/topics/[^/]+$', FLAGS.pubsub_topic):
    topic = FLAGS.pubsub_topic
  else:
    topic = f'projects/{project}/topics/{FLAGS.pubsub_topic}'
  publisher = pubsub_v1.PublisherClient()

  while True:
    try:
      logging.info("Listening for completed jobs...")
      k8s_client = kubernetes.client.BatchV1Api()
      job_watcher = kubernetes.watch.Watch()
      event_stream = job_watcher.stream(
          k8s_client.list_namespaced_job,
          FLAGS.namespace,
          label_selector='benchmarkId',
          resource_version=resource_version)
      for event in event_stream:
        _update_health()
        job = event['object']
        if event['type'] != 'MODIFIED':
          logging.info('Skipping event %s for Job %s', event['type'], job.metadata.name)
          continue
        elif job.status.active or not job.status.conditions:
          logging.info('Job %s still active or has no conditions: %s', job.metadata.name, str(job.status))
          continue

        try:
          message = create_test_completed_event(job, FLAGS.model_output_bucket, cluster_name, cluster_location, project)
        except Exception as e:
          logging.error('Error while processing job {}'.format(job), exc_info=e)
          message = None

        if message:
          publisher.publish(topic, message.SerializeToString())
          current_version = job.metadata.resource_version
          _save_resource_version(current_version)
          resource_version = current_version
          logging.info('Published message for %s:\n%s', job.metadata.name, str(message))
        else:
          logging.info('No message to publish for %s.', job.metadata.name)
    except kubernetes.client.ApiException as e:
      if e.status == 410:
        # Try to parse current resourceVersion from error message.
        match = re.fullmatch(r'Expired: too old resource version: \d+ \((\d+)\)', e.reason)
        if not match:
          logging.error('Failed to parse current resource version from message.', exc_info=e)
          raise e

        current_version = match.group(1)
        logging.warning('Resource version %d too old. Trying resource version %s.', resource_version, current_version)
        resource_version = current_version

if __name__ == '__main__':
  flags.mark_flag_as_required('model_output_bucket')
  flags.mark_flag_as_required('pubsub_topic')
  app.run(main)
