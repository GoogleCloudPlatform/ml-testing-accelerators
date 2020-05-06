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

# TODO: Consider a templating engine like Jinja to generate these strings.

BASE_MODAL_STRING = """
    var css_style = `
      body {font-family: Arial, Helvetica, sans-serif;}
      .modal {
        display: none;
        position: fixed;
        z-index: 1;
        padding-top: 100px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgb(0,0,0);
        background-color: rgba(0,0,0,0.4);
      }
      .modal-content {
        background-color: #fefefe;
        margin: auto;
        padding: 20px;
        border: 1px solid #888;
        width: 40%;
      }
      .close {
        color: #aaaaaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
      }
      .close:hover,
      .close:focus {
        color: #000;
        text-decoration: none;
        cursor: pointer;
      }
    `;
    var css_ele = document.getElementById("modal_css");
    if (css_ele == null) {
      css_ele = document.createElement("style");
      css_ele.setAttribute("id", "modal_css");
      css_ele.type = "text/css";
      css_ele.innerText = css_style;
      document.head.appendChild(css_ele);
    }
    var modal_ele = document.getElementById("modal_div");
    if (modal_ele == null) {
      modal_ele = document.createElement("div");
      modal_ele.setAttribute("id", "modal_div");
      document.body.appendChild(modal_ele);
    }
"""

PASS_FAIL_GRID_MODAL_STRING = """
    var test_name = source.data['test_name'][cb_data.source.selected.indices];
    var run_date = source.data['run_date'][cb_data.source.selected.indices];
    var job_status = source.data['job_status'][cb_data.source.selected.indices];
    var failed_metrics = source.data['failed_metrics'][cb_data.source.selected.indices];
    var failed_metrics_html = '';
    if (failed_metrics.length > 0) {
      failed_metrics_html += '<ul>';
      for (var i = 0; i < failed_metrics.length; i++) {
        failed_metrics_html += `<li>${failed_metrics[i]}</li>`;
      }
      failed_metrics_html += '</ul>';
    }
    var modal_header = `<p>Test:<br/><b>${test_name}</b><br/><br/>Completion date:<br/><b>${run_date}</b><br/><br/>Job Status:<br/><b>${job_status}</b><br/></p>`;
    if (failed_metrics_html.length > 0) {
      modal_header += `<p>Out of bounds metrics:<br/>${failed_metrics_html}<br/></p>`
    }
    var metrics_link = source.data['metrics_link'][cb_data.source.selected.indices];
    var metrics_link_html = '';
    if (metrics_link.length > 0) {
      metrics_link_html = `<a href="${metrics_link}" target="_blank">Metrics History</a><br/><br/>`;
    }
    var link_to_logs = source.data['logs_link'][cb_data.source.selected.indices];
    var logs_link_html_element = `<a href="${link_to_logs}" target="_blank">Logs in Stackdriver</a>`;
    var workload_link = source.data['workload_link'][cb_data.source.selected.indices];
    var workload_link_html_element = '';

    // String will be 'NaN' if no link exists.
    if (workload_link.length > 3) {
      workload_link_html_element = `<br/><br/><a href="${workload_link}" target="_blank">Kubernetes Workload</a>`;
    }

    var logs_download_html = '';
    var logs_download_command = source.data['logs_download_command'][cb_data.source.selected.indices];
    // String will be 'NaN' if no command exists.
    if (logs_download_command.length > 3) {
      var text_to_copy = `<input type="text" value="${logs_download_command}" id="textToCopy">`;
      var copy_button = `<button onclick="function copyToClipboard(){var copyText = document.getElementById('textToCopy');copyText.select();copyText.setSelectionRange(0, 9999);document.execCommand('copy');} copyToClipboard();">Copy to clipboard</button>`;
      logs_download_html = `<br/><br/><p>Command to download full logs:</p>${text_to_copy} &nbsp ${copy_button}`;
    }
    modal_ele.innerHTML=`
      <div id="myModal" class="modal">
        <div class="modal-content">
          <span class="close">&times;</span>
          ${modal_header}
          ${metrics_link_html}
          ${logs_link_html_element}
          ${workload_link_html_element}
          ${logs_download_html}
        </div>
      </div>
    `;
    var modal = document.getElementById("myModal");
    var span = document.getElementsByClassName("close")[0];
    span.onclick = function() { modal.style.display = "none"; }
    modal.style.display = "block";
"""

METRICS_HISTORY_MODAL_STRING = """
    var test_name = source.data['test_name'][cb_data.source.selected.indices];
    var metric_name = source.data['metric_name'][cb_data.source.selected.indices];
    var metric_value = source.data['metric_value'][cb_data.source.selected.indices];
    var metric_lower_bound = source.data['metric_lower_bound'][cb_data.source.selected.indices];
    var metric_upper_bound = source.data['metric_upper_bound'][cb_data.source.selected.indices];
    var job_status = source.data['job_status'][cb_data.source.selected.indices];
    var run_date = source.data['run_date'][cb_data.source.selected.indices];
    var modal_header = `<p>Test:<br/><b>${test_name}</b><br/><br/>Metric Name:<br/><b>${metric_name}</b><br/><br/>Completion date:<br/><b>${run_date}</b><br/><br/>Job Status:<br/><b>${job_status}</b><br/><br/>Metric Value:&nbsp<b>${metric_value}</b><br/>Metric Lower Bound:&nbsp<b>${metric_lower_bound}</b><br/>Metric Upper Bound:&nbsp<b>${metric_upper_bound}</b></p>`;
    var link_to_logs = source.data['logs_link'][cb_data.source.selected.indices];
    var logs_link_html_element = `<a href="${link_to_logs}" target="_blank">Logs in Stackdriver</a>`;

    var logs_download_html = '';
    var logs_download_command = source.data['logs_download_command'][cb_data.source.selected.indices];
    // String will be 'NaN' if no command exists.
    if (logs_download_command.length > 3) {
      var text_to_copy = `<input type="text" value="${logs_download_command}" id="textToCopy">`;
      var copy_button = `<button onclick="function copyToClipboard(){var copyText = document.getElementById('textToCopy');copyText.select();copyText.setSelectionRange(0, 9999);document.execCommand('copy');} copyToClipboard();">Copy to clipboard</button>`;
      logs_download_html = `<br/><br/><p>Command to download full logs:</p>${text_to_copy} &nbsp ${copy_button}`;
    }
    modal_ele.innerHTML=`
      <div id="myModal" class="modal">
        <div class="modal-content">
          <span class="close">&times;</span>
          ${modal_header}
          ${logs_link_html_element}
          ${logs_download_html}
        </div>
      </div>
    `;
    var modal = document.getElementById("myModal");
    var span = document.getElementsByClassName("close")[0];
    span.onclick = function() { modal.style.display = "none"; }
    modal.style.display = "block";
"""


def get_modal_javascript(modal_type):
  if modal_type == 'pass_fail_grid':
    return BASE_MODAL_STRING + PASS_FAIL_GRID_MODAL_STRING
  elif modal_type == 'metrics_history':
    return BASE_MODAL_STRING + METRICS_HISTORY_MODAL_STRING
  else:
    raise ValueError('Invalid modal_type: {}'.format(modal_type))
