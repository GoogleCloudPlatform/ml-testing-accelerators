# Copyright 2021 Google LLC
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
import absl.app
import absl.flags
import datetime
import os
import flask

# Temporary workaround to import code from the old dashboard
import sys
sys.path.append('../../../dashboard')
from main_heatmap import fetch_data


absl.flags.DEFINE_string('host', '127.0.0.1', 'Address to run server on')
absl.flags.DEFINE_integer('port', 8000, 'Port to run server on')

TEST_NAME_PREFIXES = os.environ.get('TEST_NAME_PREFIXES', '').split(',')
DASHBOARD_V2_PATH = os.environ.get('DASHBOARD_V2_PATH', os.getcwd())


def create_app():
  app = flask.Flask(
      __name__,
      root_path=DASHBOARD_V2_PATH,
      static_url_path='/static',
  )

  # Main dashboard page
  @app.route('/', methods=['GET'])
  def get_dashboard_page():
    return flask.render_template('dashboard.html')

  # API for getting the list of test prefixes
  @app.route('/api/get_test_prefixes', methods=['POST'])
  def get_test_prefixes():
    return flask.jsonify(TEST_NAME_PREFIXES)

  # API for getting the heatmap data
  @app.route('/api/get_heatmap_data', methods=['POST'])
  def get_heatmap_data():
    test_name_prefix = flask.request.get_json()['test_name_prefix']
    cutoff_timestamp = (datetime.datetime.now() - datetime.timedelta(
        days=30)).strftime('%Y-%m-%d %H:%M:%S UTC')
    data = fetch_data(test_name_prefix, cutoff_timestamp)
    return data.to_json(), 200

  return app


# Entry point for running the server in development mode
def main(_):
  app = create_app()
  app.run(
      host=absl.flags.FLAGS.host,
      port=absl.flags.FLAGS.port,
      debug=True,
  )


if __name__ == '__main__':
  absl.app.run(main)
