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
import re
import socket
from datetime import datetime

from absl import logging
import google.auth
import pandas as pd
import pandas_gbq
import redis

redis_host = os.environ.get('REDISHOST', 'localhost')
redis_port = int(os.environ.get('REDISPORT', 6379))
try:
  redis_client = redis.StrictRedis(host=redis_host, port=redis_port)
  redis_client.ping()
except Exception as e:
  logging.error('Error connecting to redis instance: {}'.format(e))
  redis_client = None


def _run(query, config={}):
  return pd.read_gbq(
      query,
      project_id=google.auth.default()[1],
      dialect='standard',
      configuration=config)


def run_query(query, cache_key, config={}, expire=3600):
  if not redis_client:
    logging.error('\n\n\nno redis_client\n\n\n')
    return _run(query, config=config)
  else:
    json = redis_client.get(cache_key)
    if json is not None:
      df = pd.read_json(json, orient='records')
    else:
      df = _run(query, config=config)
      redis_client.set(cache_key, df.to_json(orient='records'), ex=expire)
  return df

