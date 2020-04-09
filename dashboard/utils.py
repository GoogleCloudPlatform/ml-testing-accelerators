import os
import re
import socket
from datetime import datetime

from absl import logging
import pandas as pd
from pymemcache.client.hash import HashClient
import pandas_gbq

####
import base64
import json
from google.oauth2 import service_account
import pandas_gbq

logging.error('MORE CREDS:')
logging.error(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
logging.error(type(os.environ['GOOGLE_APPLICATION_CREDENTIALS']))
#logging.error(json.loads(base64.b64decode(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])))
credentials = service_account.Credentials.from_service_account_info(json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS']))
####

class MemcachedDiscovery:

    def __init__(self, host='memcached.default.svc.cluster.local', port=11211, resync_interval=10):
        self._client = None
        self._t0 = None
        self._ips = []
        self.resync_interval = resync_interval
        self.host = host
        self.port = port

    def _resync(self):
        """
        Check if the list of available nodes has changed. If any change is
        detected, a new HashClient pointing to all currently available
        nodes is returned, otherwise the current client is returned.
        """
        # Collect the all Memcached pods' IP addresses
        try:
            _, _, ips = socket.gethostbyname_ex(self.host)
        except socket.gaierror:
            # The host could not be found. This mean that either the service is
            # down or that no pods are running
            ips = []
        if set(ips) != set(self._ips):
            # A different list of ips has been detected, so we generate
            # a new client
            self._ips = ips
            if self._ips:
                servers = [(ip, self.port) for ip in self._ips]
                self._client = HashClient(servers, use_pooling=True)
            else:
                self._client = None

    def get_client(self):
        # Check if we are due for a resync of Memcached nodes
        now = datetime.now()
        due_for_resync = self._t0 is None or (now - self._t0).total_seconds() > self.resync_interval
        if due_for_resync:
            # Request a resync
            self._resync()
            # Reset the timer until the next resync
            self._t0 = now
        return self._client


def _run(query):
    logging.error("HERE ARE THE ENV VARs:")
    #logging.error(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    logging.error(os.environ['GOOGLE_PROJECT_ID'])
    return pd.read_gbq(
        query,
        project_id=os.environ['GOOGLE_PROJECT_ID'],
        credentials=credentials,
        dialect='standard'
    )


def run_query(query, cache_key, expire=3600):
    memcached_client = memcached_discovery.get_client()
    if memcached_client is None:
        return _run(query)
    else:
        json = memcached_client.get(cache_key)
        if json is not None:
            df = pd.read_json(json, orient='records')
        else:
            df = _run(query)
            memcached_client.set(cache_key, df.to_json(orient='records'), expire=expire)
        return df


memcached_discovery = MemcachedDiscovery()


LOGS_DOWNLOAD_COMMAND = """gcloud logging read 'resource.type=k8s_container resource.labels.project_id={project} resource.labels.location={zone} resource.labels.cluster_name={cluster} resource.labels.namespace_name={namespace} resource.labels.pod_name:{pod}' --limit 10000000000000 --order asc --format 'value(textPayload)' --project={project} > {pod}_logs.txt && sed -i '/^$/d' {pod}_logs.txt"""
LOG_LINK_REGEX = re.compile('https://console\.cloud\.google\.com/logs\?project=(\S+)\&advancedFilter=resource\.type\%3Dk8s_container\%0Aresource\.labels\.project_id\%3D(?P<project>\S+)\%0Aresource\.labels\.location=(?P<zone>\S+)\%0Aresource\.labels\.cluster_name=(?P<cluster>\S+)\%0Aresource\.labels\.namespace_name=(?P<namespace>\S+)\%0Aresource\.labels\.pod_name:(?P<pod>\S+)(\&dateRangeUnbound=backwardInTime)?')
def get_download_command(logs_link):
  log_pieces = LOG_LINK_REGEX.match(logs_link)
  if not log_pieces:
    print('Could not parse log link to make download link. Logs link '
          'was: {}'.format(logs_link))
    return ''
  download_command = LOGS_DOWNLOAD_COMMAND.format(**log_pieces.groupdict())
  return download_command


