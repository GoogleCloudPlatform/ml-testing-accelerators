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

r'''
Simple script that launches GKE pods to act as client workers for a TPU pod.

This script strongly assumes that it is running in the context of another GKE
pod that had a TPU attached. As such, this script expects that you will provide
the current pod's name, UID, and TPU pod addresses via the downward API (the
TPU addresses are automatically given in $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS).
This script does not clean up created resources. Instead, it sets
`metadata.ownerReferences` such that GKE's garbage collector will clean up
the created pods and services when the invoking pod is deleted.

Example:

```
python3 launch_k8s_workers.py \
    --name=pytorch-xla-pods \
    --image=gcr.io/xl-ml-test/pytorch-xla:nightly \
    --owner_name=$POD_NAME \
    --owner_uid=$POD_UID \
    --tpu=$KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS \
    --cpu=4 \
    --memory=4Gi \
    -- \
    python3 /pytorch/xla/test/test_train_mp_imagenet.py --fake_data
```
'''

import concurrent.futures
import os
import random
import re
import string
import time

from absl import app
from absl import flags
from absl import logging
import kubernetes

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None,
                    'Name of worker StatefulSet. Must be unique in `namespace`.')
flags.DEFINE_string('command', None, 'Command to run on each worker.')
flags.DEFINE_string('namespace', 'default',
                    'The namespace of the created StatefulSet.')
flags.DEFINE_string('tpu', os.getenv('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS', None),
                    'List of grpc:// addresses for the TPU. Defaults to '
                    '$KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS.')
flags.DEFINE_string('owner_name', None, 'Name of Pod that owns workers, if any.')
flags.DEFINE_string('owner_uid', None, 'UUID of Pod that owns workers, if any.')
flags.DEFINE_string('image', 'gcr.io/tpu-pytorch/xla:nightly',
                    'Docker image used for workers in created StatefulSet.')
flags.DEFINE_string('cpu', '4', 'CPU request for each worker.')
flags.DEFINE_string('memory', '4Gi', 'Memory request for each worker.')
flags.DEFINE_list('volumes', None,
                  'Comma-separated list of [PVC_NAME]:[MOUNT_DIR], where '
                  '[PVC_NAME] is the name of a Kubernetes PersistentVolumeClaim '
                  'and [MOUNT_PATH] is the directory where the PVC will be '
                  'mounted.')

def _format_env(envs):
  return [{'name': k, 'value': v} for k, v in envs.items()]

# Name must consist of lower case alphanumeric characters or '-'.
def _sanitize_job_name(name):
  return re.sub(r'[^a-z0-9\-]', '-', name.lower())

def main(argv):
  if FLAGS.command and len(argv) > 1:
    logging.warning('`--command` defined. Ignoring positional arguments.')
  elif not FLAGS.command and len(argv) > 1:
    FLAGS.command = ' '.join(argv[1:])
  elif not FLAGS.command:
    logging.error(
        'Must define `--command` or give command as a positional argument.')
    return 1

  logging.info('Command to distribute: `%s`', FLAGS.command)

  try:
    kubernetes.config.load_incluster_config()
  except:
    logging.warning('No Kubernetes cluster config. Using local kube config.')
    kubernetes.config.load_kube_config()

  k8s_client = kubernetes.client.CoreV1Api()

  random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
  job_name = '{}-{}'.format(FLAGS.name, random_suffix)

  if FLAGS.owner_name:
    ownerReferences = [{
      'apiVersion': 'v1',
      'controller': True,
      'blockOwnerDeletion': True,
      'kind': 'Pod',
      'name': FLAGS.owner_name,
      'uid': FLAGS.owner_uid
    }]
  else:
    ownerReferences = None

  service_request = kubernetes.client.V1Service(**{
    'metadata': {
      'name': _sanitize_job_name(job_name),
      'ownerReferences': ownerReferences,
    },
    'spec': {
      'ports': [{
        'name': 'xrt-mesh',
        'port': 8477,
        'protocol': 'UDP',
      }],
      # Use headless service --  a load balancer is unnecessary for one pod.
      'clusterIP': 'None',
      # Bind to the master pod (i.e. index 0).
      'selector': {
        'app': 'pytorch-xla',
        'group': job_name,
        'role': 'xrt-worker',
        'index': '0'
      }
    }
  })
  service = k8s_client.create_namespaced_service(FLAGS.namespace, service_request)
  service_name = service.metadata.name

  tpu_hosts = FLAGS.tpu.split(',')
  num_workers = len(tpu_hosts)

  master_envs = {
    'XRT_TPU_CONFIG': '|'.join(
      'c_tpu_worker;{};{}'.format(i, host.replace('grpc://', ''))
      for i, host in enumerate(tpu_hosts)
    )
  }
  common_envs = {
    'XRT_LOCAL_WORKER': 'c_tpu_worker:$(INDEX)',
    'XRT_SHARD_ORDINAL': '$(INDEX)',
    'XRT_MESH_SERVICE_ADDRESS': '{}.{}.svc.cluster.local:8477'.format(
        service_name, FLAGS.namespace),
    'XRT_SHARD_WORLD_SIZE': str(num_workers),
    'TPU_NUM_DEVICES': '8',
  }

  if FLAGS.volumes:
    volumes = {
      name: mount_path for name, mount_path in
      [v.split(':') for v in FLAGS.volumes]
    }
  else:
    volumes = {}

  pods = []
  for i in range(num_workers):
    body = kubernetes.client.V1Pod(**{
      'metadata': {
        'name': f'{job_name}-{i}',
        'ownerReferences': ownerReferences,
        'labels': {
          'app': 'pytorch-xla',
          'group': job_name,
          'index': str(i),
          'role': 'xrt-worker',
        }
      },
      'spec': {
        'restartPolicy': 'Never',
        'imagePullPolicy': 'Always',
        'containers': [{
          'name': 'main',
          'image': FLAGS.image,
          'command': [
            'bash',
            '-c',
            # TODO: Implement a readiness check instead of sleeping.
            f'set -x\nsleep 10\n{FLAGS.command}'
          ],
          'env': [
            {
              'name': 'INDEX',
              'value': str(i),
            },
            *_format_env(common_envs)
          ],
          'ports': [{
            'name': 'xrt-mesh',
            'containerPort': 8477,
            'protocol': 'UDP',
          }],
          'resources': {
            'requests': {
              'cpu': FLAGS.cpu,
              'memory': FLAGS.memory,
            }
          },
          'volumeMounts': [
            {
              'name': 'dshm',
              'mountPath': '/dev/shm/',
            },
            *[
              {
                'name': name,
                'mountPath': mount_path,
              } for name, mount_path in volumes.items()
            ],
          ],
        }],
        'volumes': [
          {
            'name': 'dshm',
            'emptyDir': { 'medium': 'Memory' },
          },
          *[
            {
              'name': name,
              'persistentVolumeClaim': { 'claimName': name }
            } for name, _ in volumes.items()
          ],
        ],
      }
    })

    if i == 0:
      body.spec['containers'][0]['env'].extend(_format_env(master_envs))

    pods.append(body)

  def _watch_pod(name, namespace):
    logs_watcher = kubernetes.watch.Watch()

    while True:
      logging.info('Waiting for pod %s to start...', name)
      pod_watcher = kubernetes.watch.Watch()
      for event in pod_watcher.stream(k8s_client.list_namespaced_pod, namespace,
                                      field_selector=f'metadata.name={name}'):
        status = event['object'].status
        logging.info('Pod %s status: %s', event['object'].metadata.name, status.phase)
        if status.phase != 'Pending':
          break

      if status.container_statuses:
        container_status = status.container_statuses[0]
        if status.container_statuses[0].state.terminated:
          exit_code = container_status.state.terminated.exit_code
          if exit_code:
            logging.error('Pod %s had non-zero exit code %d', name, exit_code)

          return exit_code

      logging.info('Streaming pod logs for %s...', name)
      for line in logs_watcher.stream(k8s_client.read_namespaced_pod_log,
                                      name, namespace, _request_timeout=3600):
        logging.info('%s] %s', name, line)

      logging.warning('Lost logs stream for %s.', name)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for pod in pods:
      resp = k8s_client.create_namespaced_pod(FLAGS.namespace, pod)
      f = executor.submit(_watch_pod, resp.metadata.name, resp.metadata.namespace)
      futures.append(f)

    # Wait for pods to complete, and exit with the first non-zero exit code.
    for f in concurrent.futures.as_completed(futures):
      exit_code = f.result()
      if exit_code:
        return exit_code

if __name__ == '__main__':
  flags.mark_flags_as_required(['name', 'tpu'])
  app.run(main)
