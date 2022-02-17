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

import collections
import datetime
import logging
import typing

import jinja2
from sendgrid.helpers import mail
import pytz

template = jinja2.Template("""
<h1>New errors in {{ benchmark_id|e }}:</h1>
<ul>
  {% for message in messages %}
  <li>{{ message|e }}</li>
  {% endfor %}
</ul>
{% if debug_info %}
<h2>Debug info:</h2>
<ul>
  {% if debug_info.logs_link  %}
  <li><a href="{{ debug_info.logs_link }}">Logs link</a></li>
  {% endif %}
  {% if debug_info.details_link  %}
  <li><a href="{{ debug_info.details_link }}">Workload link</a></li>
  {% endif %}
</ul>
{% endif %}
""")


class AlertHandler(logging.Handler):
  def __init__(self, project_id, benchmark_id, debug_info, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._project_id = project_id
    self._benchmark_id = benchmark_id
    self._debug_info = debug_info

    self._records = []

  def emit(self, record):
    self._records.append(record)

  @property
  def has_errors(self):
    return bool(self._records)

  def generate_email_content(self) -> typing.Tuple[mail.Subject, mail.HtmlContent]:
    subject = mail.Subject('Errors in {} at {}'.format(
        self._benchmark_id,
        datetime.datetime.now(pytz.timezone('US/Pacific')).strftime(
            "%Y/%m/%d %H:%M:%S")))
    
    html_message_body = template.render(
      benchmark_id=self._benchmark_id,
      messages=(record.getMessage() for record in self._records),
      debug_info=self._debug_info)
    body = mail.HtmlContent(html_message_body)

    return subject, body
