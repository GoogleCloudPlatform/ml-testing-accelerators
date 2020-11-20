import re

from absl import app
from absl import flags
from absl import logging
import kubernetes

from google.protobuf import json_format
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
import metrics_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('namespace', 'default', 'TODO')
flags.DEFINE_integer('resource_version', None, 'TODO')


def _save_resource_version(resource_version):
  logging.debug('Saving resource_version %s', resource_version)
  # In practice, this will probably be a Kubernetes annotation or similar.
  with open('resource_version.txt', 'w') as f:
    f.write(resource_version)

def _load_metric_collection_config():
  # Probably also an annotation. Maybe in config map?
  with open('metrics.json') as f:
    metric_config = metrics_pb2.MetricCollectionConfig()
    json_format.Parse(f.read(), metric_config)

  return metric_config

def _watch_and_publish_job_events(namespace, resource_version=None):
  k8s_client = kubernetes.client.BatchV1Api()
  job_watcher = kubernetes.watch.Watch()
  for event in job_watcher.stream(k8s_client.list_namespaced_job, namespace, resource_version=resource_version):
    job = event['object']
    if event['type'] != 'MODIFIED':
      logging.debug('Skipping event %s for Job %s', event['type'], job.metadata.name)
      continue
    elif job.status.active or not job.status.conditions:
      logging.debug('Job %s still active or has no conditions: %s', job.metadata.name, str(job.status))
      continue

    if len(job.status.conditions) == 1:
      condition = job.status.conditions[0]
    # job.status.condiations _usually_ has length 1, but it can have both passing and failing conditions.
    # Give precedence to failing conditions.
    else:
      condition = next((c for c in job.status.conditions if c.type == 'Failed'), None)

    if condition and condition.reason == 'DeadlineExceeded':
      job_status = metrics_pb2.TestCompletedEvent.TIMEOUT
    elif condition and condition.reason == 'BackoffLimitExceeded':
      job_status = metrics_pb2.TestCompletedEvent.FAILED
    elif condition:
      job_status = metrics_pb2.TestCompletedEvent.COMPLETED
    else:
      logging.error('Unknown condition for Job %s: %s', job.metadata.name, str(condition))
      continue

    metric_config = _load_metric_collection_config()

    start_time = timestamp_pb2.Timestamp()
    start_time.FromDatetime(job.status.start_time)
    duration = duration_pb2.Duration()
    duration.FromTimedelta(condition.last_transition_time - job.status.start_time)
    tce = metrics_pb2.TestCompletedEvent(
      benchmark_id=job.metadata.name[:job.metadata.name.rfind('-')],
      status=job_status,
      start_time=start_time,
      duration=duration,
      metric_collection_config=metric_config,
    )

    _save_resource_version(event['object'].metadata.resource_version)
    print(event['type'], event['object'].metadata.resource_version, event['object'].metadata.name, tce, event['object'].status)

def main(argv):
  try:
    kubernetes.config.load_incluster_config()
  except:
    logging.warning("Using local kube config.")
    kubernetes.config.load_kube_config()

  try:
    _watch_and_publish_job_events(FLAGS.namespace, FLAGS.resource_version)
  except kubernetes.client.ApiException as e:
    if e.status == 410:
      # Try to parse current resourceVersion from error message.
      match = re.fullmatch(r'Expired: too old resource version: \d+ \((\d+)\)', e.reason)
      if not match:
        logging.error('Failed to parse current resource version from message: "%s"', e.reason)
        raise e

      current_resource_version = match.group(1)
      logging.warning('Resource version %d too old. Trying resource version %s.', FLAGS.resource_version, current_resource_version)
      _watch_and_publish_job_events(FLAGS.namespace, current_resource_version)

if __name__ == '__main__':
  logging.set_verbosity(logging.DEBUG)
  app.run(main)
