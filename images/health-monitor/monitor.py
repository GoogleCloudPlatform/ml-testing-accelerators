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

import os
import time

import cloud_tpu_client

from absl import app
from absl import flags
from absl import logging
from kubernetes import client, config

FLAGS = flags.FLAGS

flags.DEFINE_string('pod', None, 'The name of the pod to monitor.')
flags.DEFINE_string('namespace', 'default', 'The namespace of the pod.')
flags.DEFINE_string('container', 'train', 'The name of the container to watch.')

flags.DEFINE_integer('interval', 60, 'Number of seconds to wait between health checks.')
flags.DEFINE_bool('verbose', False, 'Whether to print when TPU is HEALTHY.')

flags.DEFINE_string('project', None, 'The GCP project with your GKE cluster.')
flags.DEFINE_string('zone', None, 'The GCP zone with your GKE cluster.')

def main(_):
  if FLAGS.verbose:
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.WARNING)

  try:
    config.load_incluster_config()
  except config.ConfigException:
    config.load_kube_config()

  k8s_client = client.CoreV1Api()
  pod = k8s_client.read_namespaced_pod(FLAGS.pod, FLAGS.namespace)

  tpu_name_annotation = 'name.cloud-tpus.google.com/{}'.format(FLAGS.container)
  tpu_name = os.path.basename(pod.metadata.annotations[tpu_name_annotation])

  tpu_client = cloud_tpu_client.Client(tpu_name, FLAGS.zone, FLAGS.project)

  while True:
    try:
      health = tpu_client.health()
    except ValueError as e:
      logging.error('Error getting TPU status: %s', str(e))
      health = None

    if health == 'HEALTHY':
      logging.info('TPU health: %s', health)
    else:
      logging.warning('TPU health: %s', health)

      if not tpu_client.recoverable():
        logging.warning('TPU entered un-recoverable state: %s',
                        tpu_client.state())
        break

    pod = k8s_client.read_namespaced_pod_status(FLAGS.pod, FLAGS.namespace)
    try:
      status = next(c for c in pod.status.container_statuses
                    if c.name == FLAGS.container)
    except StopIteration:
      logging.fatal('Status for container `%s` not found in statuses:\n%s',
                    FLAGS.container, str(pod.status))
      exit(1)
    if getattr(status.state, 'terminated'):
      logging.warning('Container `%s` terminated with status:\n%s',
                      FLAGS.container, str(status))
      break
  
    time.sleep(FLAGS.interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('pod')
  app.run(main)
